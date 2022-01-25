"""Soft Actor Critic."""
import argparse
from typing import List, Tuple, Dict

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import gym

import wordle
import sac
from sac.agent import SoftActorCriticAgent
from sac.experience import ExperienceSourceDataset, MultiStepBuffer, Experience
from sac.mlp import MLP


class SAC(LightningModule):
    def __init__(
        self,
        env: str,
            network_name: str,
        eps_start: float = 1.0,
        eps_end: float = 0.02,
        eps_last_frame: int = 150000,
        sync_rate: int = 1,
        gamma: float = 0.99,
        policy_learning_rate: float = 3e-4,
        q_learning_rate: float = 3e-4,
        target_alpha: float = 5e-3,
        batch_size: int = 512,
        replay_size: int = 1000,
        warm_start_size: int = 1000,
        avg_reward_len: int = 100,
        min_episode_reward: int = -10,
        seed: int = 123,
        batches_per_epoch: int = 10000,
        n_steps: int = 1,
        **kwargs,
    ):
        super().__init__()

        # Hyperparameters
        self.save_hyperparameters()

        self.writer = SummaryWriter()
        # Environment
        self.env = gym.make(env)
        self.test_env = gym.make(env)

        # Model Attributes
        self.buffer = None
        self.dataset = None

        """Initializes the SAC policy and q networks (with targets)"""
        self.env = gym.make(env)
        self.policy = sac.construct(
            self.hparams.network_name,
            obs_size=self.env.observation_space.shape[0],
            n_hidden=self.hparams.n_hidden,
            hidden_size=self.hparams.hidden_size,
            word_list=self.env.words)

        concat_shape = self.env.observation_space.shape[0] + 26*5
        self.q1 = MLP(concat_shape, 1, word_list=self.env.words)
        self.q2 = MLP(concat_shape, 1, word_list=self.env.words)
        self.target_q1 = MLP(concat_shape, 1, word_list=self.env.words)
        self.target_q2 = MLP(concat_shape, 1, word_list=self.env.words)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.agent = SoftActorCriticAgent(self.policy)

        # Metrics
        self.total_episode_steps = [0]
        self.total_rewards = [0]
        self.done_episodes = 0
        self.total_steps = 0

        # Average Rewards
        self.avg_reward_len = avg_reward_len

        for _ in range(avg_reward_len):
            self.total_rewards.append(torch.tensor(min_episode_reward, device=self.device))

        self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len :]))

        self.state = self.env.reset()

        self.automatic_optimization = False

        self._winning_steps = 0
        self._winning_rewards = 0
        self._total_rewards = 0
        self._wins = 0
        self._losses = 0

    def run_n_episodes(self, env, n_epsiodes: int = 1) -> List[int]:
        """Carries out N episodes of the environment with the current agent without exploration.
        Args:
            env: environment to use, either train environment or test environment
            n_epsiodes: number of episodes to run
        """
        total_rewards = []

        for _ in range(n_epsiodes):
            episode_state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.agent.get_action_greedy(episode_state, self.device)
                next_state, reward, done, _ = env.step(action[0])
                episode_state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)

        return total_rewards

    def populate(self, warm_start: int) -> None:
        """Populates the buffer with initial experience."""
        if warm_start > 0:
            self.state = self.env.reset()

            for _ in range(warm_start):
                action = self.agent(self.state, self.device)
                next_state, reward, done, _ = self.env.step(action[0])
                exp = Experience(state=self.state, action=action[0], reward=reward, done=done, new_state=next_state)
                self.buffer.append(exp)
                self.state = next_state

                if done:
                    self.state = self.env.reset()

    def soft_update_target(self, q_net, target_net):
        """Update the weights in target network using a weighted sum.
        w_target := (1-a) * w_target + a * w_q
        Args:
            q_net: the critic (q) network
            target_net: the target (q) network
        """
        for q_param, target_param in zip(q_net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                (1.0 - self.hparams.target_alpha) * target_param.data + self.hparams.target_alpha * q_param
            )

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.
        Args:
            x: environment state
        Returns:
            q values
        """
        output = self.policy(x).sample()
        return output

    def train_batch(
        self,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Contains the logic for generating a new batch of data to be passed to the DataLoader.
        Returns:
            yields a Experience tuple containing the state, action, reward, done and next_state.
        """
        episode_reward = 0
        episode_steps = 0

        while True:
            self.total_steps += 1
            action = self.agent(self.state, self.device)

            next_state, r, is_done, _ = self.env.step(action[0])

            episode_reward += r
            episode_steps += 1

            exp = Experience(state=self.state,
                             action=action[0],
                             reward=r,
                             done=is_done,
                             new_state=next_state)

            self.buffer.append(exp)
            self.state = next_state

            if is_done:
                if action[0] == self.env.goal_word:
                    self._wins += 1
                    self._winning_steps += self.env.max_turns - self.state.remaining_steps()
                    self._winning_rewards += episode_reward
                else:
                    self._losses += 1

                self.done_episodes += 1
                self.total_rewards.append(episode_reward)
                self.total_episode_steps.append(episode_steps)
                self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len :]))
                self.state = self.env.reset()
                episode_steps = 0
                episode_reward = 0

            states, actions, rewards, dones, new_states = self.buffer.sample(self.hparams.batch_size)

            for idx, _ in enumerate(dones):
                yield states[idx], actions[idx], rewards[idx], dones[idx], new_states[idx]

            # Simulates epochs
            if self.total_steps % self.hparams.batches_per_epoch == 0:
                break

    def _actions_ohe(self, actions: List[int], shape: Tuple[int, int], device: str) -> torch.Tensor:
        ohe = torch.zeros(shape, dtype=torch.int32, device=device)
        for i, a in enumerate(actions):
            ohe[i, a] = 1

        return ohe

    def loss(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Calculates the loss for SAC which contains a total of 3 losses.
        Args:
            batch: a batch of states, actions, rewards, dones, and next states
        """
        states, actions, rewards, dones, next_states = batch
        rewards = rewards.unsqueeze(-1)
        dones = dones.float().unsqueeze(-1)

        # actor
        actions_shape = (states.shape[0], len(self.env.words))
        new_actions, new_logprobs = self.agent.sample_actions_and_logprobs(states, self.device)
        #new_actions_ohe = self._actions_ohe(actions=new_actions, shape=actions_shape, device=self.device)

        # dist = self.policy(states)
        # new_actions, new_logprobs = dist.rsample_and_log_prob()
        # new_logprobs = new_logprobs.unsqueeze(-1)
        #new_states_actions = torch.cat((states, new_actions_ohe), 1)
        new_q1_values = self.q1(states, new_actions)
        new_q2_values = self.q2(states, new_actions)
        new_qmin_values = torch.min(new_q1_values, new_q2_values)

        policy_loss = (new_logprobs - new_qmin_values).mean()
        #print(new_logprobs.view(1, -1).shape, new_qmin_values.shape, policy_loss)

        # critic
        #action_ohe = self._actions_ohe(actions=actions, shape=actions_shape, device=self.device)
        #states_actions = torch.cat((states, action_ohe), 1)
        q1_values = self.q1(states, actions)
        q2_values = self.q2(states, actions)

        with torch.no_grad():
            new_next_actions, new_next_logprobs = self.agent.sample_actions_and_logprobs(next_states, self.device)
            #new_next_actions_ohe = self._actions_ohe(new_next_actions, actions_shape, self.device)
            # next_dist = self.policy(next_states)
            # new_next_actions, new_next_logprobs = next_dist.rsample_and_log_prob()
            # new_next_logprobs = new_next_logprobs.unsqueeze(-1)
            #new_next_states_actions = torch.cat((next_states, new_next_actions_ohe), 1)
            next_q1_values = self.target_q1(next_states, new_next_actions)
            next_q2_values = self.target_q2(next_states, new_next_actions)
            next_qmin_values = torch.min(next_q1_values, next_q2_values) - new_next_logprobs
            target_values = rewards + (1.0 - dones) * self.hparams.gamma * next_qmin_values
            #target_values = 0.05*rewards + self.hparams.gamma * next_qmin_values

        q1_loss = F.mse_loss(q1_values, target_values)
        q2_loss = F.mse_loss(q2_values, target_values)

        return policy_loss, q1_loss, q2_loss

    def training_step(self, batch: Tuple[Tensor, Tensor], _):
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.
        Args:
            batch: current mini batch of replay data
            _: batch number, not used
        """
        policy_optim, q1_optim, q2_optim = self.optimizers()
        policy_loss, q1_loss, q2_loss = self.loss(batch)

        policy_optim.zero_grad()
        self.manual_backward(policy_loss)
        policy_optim.step()

        q1_optim.zero_grad()
        self.manual_backward(q1_loss)
        q1_optim.step()

        q2_optim.zero_grad()
        self.manual_backward(q2_loss)
        q2_optim.step()

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.soft_update_target(self.q1, self.target_q1)
            self.soft_update_target(self.q2, self.target_q2)

        if self.global_step % 100 == 0:
            # # Find a sequence
            # i = 0
            # while i < len(states):
            #     if states[i][0] == self.env.max_turns:
            #         break
            #     i += 1
            # # Walk through game
            # game = f"goal: {self.env.words[goal_ids[i]]}\n"
            # strt = i
            # game += f'{0}: {self.env.words[actions[i]]}\n'
            # i += 1
            # while i < len(states) and states[i][0] != self.env.max_turns:
            #     game += f'{i-strt}: {self.env.words[actions[i]]}\n'
            #     i += 1

            #self.writer.add_text("game sample", game, global_step=self.global_step)
            #self.writer.add_scalar("train_loss", loss, global_step=self.global_step)
            self.writer.add_scalar("total_games_played", self.done_episodes, global_step=self.global_step)

            if self._wins + self._losses > 0:
                self.writer.add_scalar("lose_ratio", self._losses / (self._wins + self._losses), global_step=self.global_step)
                self.writer.add_scalar("wins", self._wins, global_step=self.global_step)
                self.writer.add_scalar("reward_per_game", self._total_rewards / (self._wins + self._losses),
                                       global_step=self.global_step)
            if self._wins > 0:
                self.writer.add_scalar("reward_per_win", self._winning_rewards / self._wins, global_step=self.global_step)
                self.writer.add_scalar("avg_winning_turns", self._winning_steps / self._wins, global_step=self.global_step)

            self._winning_steps = 0
            self._winning_rewards = 0
            self._total_rewards = 0
            self._wins = 0
            self._losses = 0

        self.log_dict(
            {
                "total_reward": self.total_rewards[-1],
                "avg_reward": self.avg_rewards,
                "policy_loss": policy_loss,
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "episodes": self.done_episodes,
                "episode_steps": self.total_episode_steps[-1],
            }
        )

    def test_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        """Evaluate the agent for 10 episodes."""
        test_reward = self.run_n_episodes(self.test_env, 1)
        avg_reward = sum(test_reward) / len(test_reward)
        return {"test_reward": avg_reward}

    def test_epoch_end(self, outputs) -> Dict[str, Tensor]:
        """Log the avg of the test results."""
        rewards = [x["test_reward"] for x in outputs]
        avg_reward = sum(rewards) / len(rewards)
        self.log("avg_test_reward", avg_reward)
        return {"avg_test_reward": avg_reward}

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        self.buffer = MultiStepBuffer(self.hparams.replay_size, self.hparams.n_steps)
        self.populate(self.hparams.warm_start_size)

        self.dataset = ExperienceSourceDataset(self.train_batch)
        return DataLoader(dataset=self.dataset, batch_size=self.hparams.batch_size)

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self._dataloader()

    def test_dataloader(self) -> DataLoader:
        """Get test loader."""
        return self._dataloader()

    def configure_optimizers(self) -> Tuple[Optimizer]:
        """Initialize Adam optimizer."""
        policy_optim = optim.Adam(self.policy.parameters(), self.hparams.policy_learning_rate)
        q1_optim = optim.Adam(self.q1.parameters(), self.hparams.q_learning_rate)
        q2_optim = optim.Adam(self.q2.parameters(), self.hparams.q_learning_rate)
        return policy_optim, q1_optim, q2_optim

    @staticmethod
    def add_model_specific_args(
        arg_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Adds arguments for DQN model.
        Note:
            These params are fine tuned for Pong env.
        Args:
            arg_parser: parent parser
        """
        arg_parser.add_argument(
            "--sync_rate",
            type=int,
            default=1,
            help="how many frames do we update the target network",
        )
        arg_parser.add_argument(
            "--replay_size",
            type=int,
            default=1000,
            help="capacity of the replay buffer",
        )
        arg_parser.add_argument(
            "--warm_start_size",
            type=int,
            default=1000,
            help="how many samples do we use to fill our buffer at the start of training",
        )
        arg_parser.add_argument("--batches_per_epoch", type=int, default=1000, help="number of batches in an epoch")
        arg_parser.add_argument("--batch_size", type=int, default=512, help="size of the batches")
        arg_parser.add_argument("--policy_lr", type=float, default=3e-4, help="policy learning rate")
        arg_parser.add_argument("--q_lr", type=float, default=3e-4, help="q learning rate")
        arg_parser.add_argument("--env", type=str, required=True, help="gym environment tag")
        arg_parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
        arg_parser.add_argument("--network_name", type=str, default="SumChars", help="Network to use")

        arg_parser.add_argument("--n_hidden", type=int, default="1", help="Number of hidden layers")
        arg_parser.add_argument("--hidden_size", type=int, default="256", help="Width of hidden layers")
        arg_parser.add_argument("--seed", type=int, default=123, help="seed for training run")

        arg_parser.add_argument(
            "--avg_reward_len",
            type=int,
            default=100,
            help="how many episodes to include in avg reward",
        )
        arg_parser.add_argument(
            "--n_steps",
            type=int,
            default=10,
            help="how many frames do we update the target network",
        )

        return arg_parser


def cli_main():
    parser = argparse.ArgumentParser(add_help=False)

    # trainer args
    parser = Trainer.add_argparse_args(parser)

    # model args
    parser = SAC.add_model_specific_args(parser)
    args = parser.parse_args()

    model = SAC(**args.__dict__)

    # save checkpoints based on avg_reward
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="avg_reward", mode="max", verbose=True)

    seed_everything(123)
    trainer = Trainer.from_argparse_args(args, deterministic=True, callbacks=checkpoint_callback)

    trainer.fit(model)


if __name__ == "__main__":
    cli_main()