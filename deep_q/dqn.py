import os
from collections import OrderedDict, deque, namedtuple
from typing import List, Tuple

import gym
import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import DistributedType
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from deep_q.agent import Agent
from deep_q.experience import ReplayBuffer, RLDataset

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")


class DQN(nn.Module):
    """Simple MLP network."""

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 256):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x.float())


class DQNLightning(LightningModule):
    """Basic DQN Model."""

    def __init__(
        self,
        batch_size: int = 16,
        lr: float = 1e-2,
        env: str = "WordleEnv-v0",
        gamma: float = 0.9,
        sync_rate: int = 20,
        replay_size: int = 1000,
        hidden_size: int = 256,
        num_workers: int = 0,
        warm_start_size: int = 1000,
        eps_last_frame: int = 10000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        episode_length: int = 25,
        warm_start_steps: int = 1000,
    ) -> None:
        """
        Args:
            batch_size: size of the batches")
            lr: learning rate
            env: gym environment tag
            gamma: discount factor
            sync_rate: how many frames do we update the target network
            replay_size: capacity of the replay buffer
            warm_start_size: how many samples do we use to fill our buffer at the start of training
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode
            warm_start_steps: max episode reward in the environment
        """
        super().__init__()
        self.save_hyperparameters()

        self.writer = SummaryWriter()
        self.env = gym.make(self.hparams.env)
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self._winning_steps = 0
        self._wins = 0
        self._losses = 0

        print("dqn:", self.env.spec.id, self.env.spec.max_episode_steps, n_actions, obs_size)

        self.net = DQN(obs_size, n_actions, hidden_size=hidden_size)
        self.target_net = DQN(obs_size, n_actions, hidden_size=hidden_size)

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hparams.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = max(
            self.hparams.eps_end,
            self.hparams.eps_start - self.global_step / self.hparams.eps_last_frame,
        )

        # step through environment with agent
        reward, done, winning_steps = self.agent.play_step(self.net, epsilon, device)
        if reward < 0:
            self._losses += 1
        elif reward > 0:
            self._wins += 1
            self._winning_steps += winning_steps

        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if self.trainer._distrib_type in {DistributedType.DP, DistributedType.DDP2}:
            loss = loss.unsqueeze(0)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "train_loss": loss.detach(),
        }
        status = {
            "steps": torch.tensor(self.global_step).to(device),
            "total_reward": torch.tensor(self.total_reward).to(device),
        }

        if self.global_step % 100 == 0:
            self.writer.add_scalar("total_reward", self.total_reward, global_step=self.global_step)
            self.writer.add_scalar("train_loss", loss, global_step=self.global_step)
            if self._wins + self._losses > 0:
                self.writer.add_scalar("lose_ratio", self._losses/(self._wins+self._losses), global_step=self.global_step)
            if self._wins > 0:
                self.writer.add_scalar("avg_winning_turns", self._winning_steps/self._wins, global_step=self.global_step)
            self._winning_steps = 0
            self._wins = 0
            self._losses = 0

        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"
