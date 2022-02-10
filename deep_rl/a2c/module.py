import collections
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Any, List, Tuple, Iterator
import wandb

import gym
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import Tensor, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import a2c
import wordle.state
from a2c.agent import ActorCriticAgent
from a2c.experience import ExperienceSourceDataset, Experience


class AdvantageActorCritic(LightningModule):
    """PyTorch Lightning implementation of `Advantage Actor Critic <https://arxiv.org/abs/1602.01783v2>`_.
    Paper Authors: Volodymyr Mnih, Adrià Puigdomènech Badia, et al.
    Model implemented by:
        - `Jason Wang <https://github.com/blahBlahhhJ>`_
    """

    def __init__(
            self,
            env: str,
            network_name: str,
            gamma: float,
            lr: float,
            batch_size: int,
            avg_reward_len: int,
            n_hidden: int,
            hidden_size: int,
            entropy_beta: float,
            critic_beta: float,
            epoch_len: int,
            prob_play_lost_word: float=0.,
            prob_cheat: float=0.,
            weight_decay: float=0.,
            **kwargs: Any,
    ) -> None:
        """
        Args:
            env: gym environment tag
            gamma: discount factor
            lr: learning rate
            batch_size: size of minibatch pulled from the DataLoader
            batch_episodes: how many episodes to rollout for each batch of training
            avg_reward_len: how many episodes to take into account when calculating the avg reward
            entropy_beta: dictates the level of entropy per batch
            critic_beta: dictates the level of critic loss per batch
            epoch_len: how many batches before pseudo epoch
        """
        super().__init__()

        # Hyperparameters
        self.save_hyperparameters()
        self.writer = SummaryWriter()
        self.batches_per_epoch = batch_size * epoch_len

        # Model components
        self.env = gym.make(env)
        self.net = a2c.construct(
            self.hparams.network_name,
            obs_size=self.env.observation_space.shape[0],
            n_hidden=self.hparams.n_hidden,
            hidden_size=self.hparams.hidden_size,
            word_list=self.env.words)
        self.agent = ActorCriticAgent(self.net)

        # Tracking metrics
        self.episode_reward = 0
        self.done_episodes = 0
        self.eps = np.finfo(np.float32).eps.item()

        self._winning_steps = 0
        self._winning_rewards = 0
        self._total_rewards = 0
        self._wins = 0
        self._losses = 0
        self._last_win = []
        self._last_loss = []
        self._seq = []

        self._recent_losing_words = collections.deque(maxlen=1000)
        self._cheat_word = None

        self.state = self.env.reset()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Passes in a state x through the network and gets the log prob of each action and the value for the state
        as an output.
        Args:
            x: environment state
        Returns:
            action log probabilities, values
        """
        # if not isinstance(x, list):
        #     x = [x]
        #
        # if not isinstance(x, Tensor):
        #     x = torch.tensor(x, device=self.device)
        #
        logprobs, values = self.net(torch.tensor([x], device=self.device))
        return logprobs, values

    def train_batch(self) -> Iterator[Tuple[np.ndarray, int, Tensor]]:
        """Contains the logic for generating a new batch of data to be passed to the DataLoader.
        Returns:
            yields a tuple of Lists containing tensors for
            states, actions, and returns of the batch.
        Note:
            This is what's taken by the dataloader:
            states: a list of numpy array
            actions: a list of list of int
            returns: a torch tensor
        """
        while True:
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_masks = []
            batch_targets = []
            for _ in range(self.hparams.batch_size):
                action = self.agent(self.state, self.device)[0]
                if wordle.state.remaining_steps(self.state) == 1 and self._cheat_word:
                    action = self._cheat_word

                next_state, reward, done, aux = self.env.step(action)

                batch_states.append(self.state)
                batch_actions.append(action)
                batch_rewards.append(reward)
                batch_masks.append(done)
                batch_targets.append(aux['goal_id'])

                self._seq.append(Experience(self.state.copy(), action, reward, aux['goal_id']))
                self.state = next_state
                self.episode_reward += reward

                if done:
                    if action == self.env.goal_word:
                        self._winning_steps += self.env.max_turns - wordle.state.remaining_steps(self.state)
                        self._wins += 1
                        self._winning_rewards += self.episode_reward
                        self._last_win = self._seq
                    else:
                        self._losses += 1
                        self._last_loss = self._seq
                        self._recent_losing_words.append(aux['goal_id'])
                    self._seq = []
                    self._total_rewards += self.episode_reward

                    self.done_episodes += 1
                    # With some probability, override the word with one that we lost recently
                    self.state = self.env.reset()
                    self._cheat_word = None
                    if len(self._recent_losing_words) > 0:
                        if np.random.random() < self.hparams.prob_play_lost_word:
                            lost_idx = int(np.random.random()*len(self._recent_losing_words))
                            self.env.set_goal_id(self._recent_losing_words[lost_idx])
                            if np.random.random() < self.hparams.prob_cheat:
                                self._cheat_word = self._recent_losing_words[lost_idx]

                    self.episode_reward = 0

            _, last_value = self.forward(self.state)

            returns = self.compute_returns(batch_rewards, batch_masks, last_value)
            for idx in range(self.hparams.batch_size):
                yield batch_states[idx], batch_actions[idx], returns[idx], batch_targets[idx]

    def compute_returns(
            self,
            rewards: List[float],
            dones: List[bool],
            last_value: Tensor,
    ) -> Tensor:
        """Calculate the discounted rewards of the batched rewards.
        Args:
            rewards: list of rewards
            dones: list of done masks
            last_value: the predicted value for the last state (for bootstrap)
        Returns:
            tensor of discounted rewards
        """
        g = last_value
        returns = []

        for r, d in zip(rewards[::-1], dones[::-1]):
            g = r + self.hparams.gamma * g * (1 - d)
            returns.append(g)

        # reverse list and stop the gradients
        returns = torch.tensor(returns[::-1])

        return returns

    def loss(
            self,
            states: Tensor,
            actions: Tensor,
            returns: Tensor,
    ) -> Tensor:
        """Calculates the loss for A2C which is a weighted sum of actor loss (MSE), critic loss (PG), and entropy
        (for exploration)
        Args:
            states: tensor of shape (batch_size, state dimension)
            actions: tensor of shape (batch_size, )
            returns: tensor of shape (batch_size, )
        """

        logprobs, values = self.net(states)

        # calculates (normalized) advantage
        with torch.no_grad():
            # critic is trained with normalized returns, so we need to scale the values here
            advs = returns - values * returns.std() + returns.mean()
            # normalize advantages to train actor
            advs = (advs - advs.mean()) / (advs.std() + self.eps)
            # normalize returns to train critic
            targets = (returns - returns.mean()) / (returns.std() + self.eps)

        # entropy loss
        entropy = -logprobs.exp() * logprobs
        entropy = self.hparams.entropy_beta * entropy.sum(1).mean()

        # actor loss
        logprobs = logprobs[range(len(actions)), actions]
        actor_loss = -(logprobs * advs).mean()

        # critic loss
        critic_loss = self.hparams.critic_beta * torch.square(targets - values).mean()

        # total loss (weighted sum)
        total_loss = actor_loss + critic_loss - entropy
        return total_loss

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> OrderedDict:
        """Perform one actor-critic update using a batch of data.
        Args:
            batch: a batch of (states, actions, returns)
        """
        states, actions, returns, goal_ids = batch

        # Compute loss to backprop
        loss = self.loss(states, actions, returns)

        if self.global_step % 50 == 0:
            metrics = {
                "train_loss": loss,
                "total_games_played": self.done_episodes,
                "lose_ratio": self._losses/(self._wins+self._losses),
                "wins": self._wins,
                "reward_per_game": self._total_rewards / (self._wins+self._losses),
                "global_step": self.global_step,
            }
            if self._wins > 0:
                metrics["reward_per_win"] = self._winning_rewards / self._wins
                metrics["avg_winning_turns"] = self._winning_steps / self._wins

            for k, v in metrics.items():
                self.writer.add_scalar(k, v, global_step=self.global_step)

            def get_game_string(seq):
                game = f'goal: {self.env.words[seq[0].goal_id]}\n'
                for i, exp in enumerate(seq):
                    game += f'{i}: {self.env.words[exp.action]}\n'
                return game

            def get_table_row(seq):
                goal = self.env.words[seq[0].goal_id]
                guesses = ""
                for i, exp in enumerate(seq):
                    guesses += f'{i}: {self.env.words[exp.action]} '
                return [goal, guesses]

            if len(self._last_win):
                self.writer.add_text("last_win", get_game_string(self._last_win), global_step=self.global_step)
                metrics["last_win"] = wandb.Table(data=[get_table_row(self._last_win)], columns=['goal', 'guesses'])
            if len(self._last_loss):
                self.writer.add_text("last_loss", get_game_string(self._last_loss), global_step=self.global_step)
                metrics["last_loss"] = wandb.Table(data=[get_table_row(self._last_loss)], columns=['goal', 'guesses'])

            wandb.log(metrics)
            # self.writer.add_scalar("train_loss", loss, global_step=self.global_step)
            # self.writer.add_scalar("total_games_played", self.done_episodes, global_step=self.global_step)
            #
            # self.writer.add_scalar("lose_ratio", self._losses/(self._wins+self._losses), global_step=self.global_step)
            # self.writer.add_scalar("wins", self._wins, global_step=self.global_step)
            # self.writer.add_scalar("reward_per_game", self._total_rewards / (self._wins+self._losses), global_step=self.global_step)
            # if self._wins > 0:
            #     self.writer.add_scalar("reward_per_win", self._winning_rewards / self._wins, global_step=self.global_step)
            #     self.writer.add_scalar("avg_winning_turns", self._winning_steps/self._wins, global_step=self.global_step)

            self._winning_steps = 0
            self._winning_rewards = 0
            self._total_rewards = 0
            self._wins = 0
            self._losses = 0

        log = {
            "episodes": self.done_episodes,
        }
        return OrderedDict(
            {
                "loss": loss,
                "log": log,
                "progress_bar": log,
            }
        )

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = optim.Adam(self.net.parameters(),
                               lr=self.hparams.lr,
                               weight_decay=self.hparams.weight_decay)
        return [optimizer]

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = ExperienceSourceDataset(self.train_batch)
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self._dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0][0][0].device.index if self.on_gpu else "cpu"

    @staticmethod
    def add_model_specific_args(arg_parser: ArgumentParser) -> ArgumentParser:
        """Adds arguments for A2C model.
        Args:
            arg_parser: the current argument parser to add to
        Returns:
            arg_parser with model specific cargs added
        """

        arg_parser.add_argument("--entropy_beta", type=float, default=0.01, help="entropy coefficient")
        arg_parser.add_argument("--critic_beta", type=float, default=0.5, help="critic loss coefficient")
        arg_parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
        arg_parser.add_argument("--epoch_len", type=int, default=10, help="Batches per epoch")
        arg_parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        arg_parser.add_argument("--env", type=str, default="WordleEnv100-v0", help="gym environment tag")
        arg_parser.add_argument("--network_name", type=str, default="SumChars", help="Network to use")
        arg_parser.add_argument("--n_hidden", type=int, default="1", help="Number of hidden layers")
        arg_parser.add_argument("--hidden_size", type=int, default="256", help="Width of hidden layers")
        arg_parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
        arg_parser.add_argument("--seed", type=int, default=123, help="seed for training run")
        arg_parser.add_argument("--replay_size", type=int, default=1000, help="Size of replay buffer(s)")
        arg_parser.add_argument("--prob_play_lost_word", type=float, default=0, help="Probabiilty of replaying a losing word")
        arg_parser.add_argument("--prob_cheat", type=float, default=0, help="Probability of cheating when playing lost word")
        arg_parser.add_argument("--weight_decay", type=float, default=0., help="Optimizer weight decay regularization.")

        arg_parser.add_argument(
            "--avg_reward_len",
            type=int,
            default=100,
            help="how many episodes to include in avg reward",
        )

        return arg_parser