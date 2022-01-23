"""Advantage Actor Critic (A2C)"""
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Any, Iterator, List, Tuple

import numpy as np
import gym
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor, optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import a2c
from a2c.agent import ActorCriticAgent
from a2c.experience import ExperienceSourceDataset, RLDataset, SequenceReplay, Experience


class AdvantageActorCritic(LightningModule):
    """PyTorch Lightning implementation of `Advantage Actor Critic <https://arxiv.org/abs/1602.01783v2>`_.
    Paper Authors: Volodymyr Mnih, Adrià Puigdomènech Badia, et al.
    Model implemented by:
        - `Jason Wang <https://github.com/blahBlahhhJ>`_
    Example:
        >>> from pl_bolts.models.rl import AdvantageActorCritic
        ...
        >>> model = AdvantageActorCritic("CartPole-v0")
    """

    def __init__(
            self,
            env: str,
            network_name: str,
            gamma: float,
            lr: float,
            batch_size: int,
            avg_reward_len: int,
            hidden_size: int,
            entropy_beta: float,
            critic_beta: float,
            epoch_len: int,
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
            n_actions=self.env.action_space.n,
            hidden_size=self.hparams.hidden_size,
            word_list=self.env.words)
        self.agent = ActorCriticAgent(self.net)

        self.dataset = RLDataset(
            winners=SequenceReplay(self.hparams.replay_size//2),# self.hparams.initialize_winning_replays),
            losers=SequenceReplay(self.hparams.replay_size//2),
            sample_size=self.hparams.batch_size)

        # Tracking metrics
        self.total_rewards = [0]
        self.episode_reward = 0
        self.done_episodes = 0
        self.avg_rewards = 0.0
        self.avg_reward_len = avg_reward_len
        self.eps = np.finfo(np.float32).eps.item()
        self.batch_states: List = []
        self.batch_actions: List = []
        self.batch_rewards: List = []
        self.batch_masks: List = []
        self.batch_targets: List = []

        self._winning_steps = 0
        self._winning_rewards = 0
        self._total_rewards = 0
        self._wins = 0
        self._losses = 0

        self.state = self.env.reset()

        # Populate with some samples
        # for _ in range(1000):
        #     self.play_game()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Passes in a state x through the network and gets the log prob of each action and the value for the state
        as an output.
        Args:
            x: environment state
        Returns:
            action log probabilities, values
        """
        if not isinstance(x, list):
            x = [x]

        if not isinstance(x, Tensor):
            x = torch.tensor(x, device=self.device)

        logprobs, values = self.net(x)
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
            for _ in range(self.hparams.batch_size):
                action = self.agent(self.state.vec, self.device)[0]

                next_state, reward, done, aux = self.env.step(action)

                self.batch_rewards.append(reward)
                self.batch_actions.append(action)
                self.batch_states.append(self.state.vec)
                self.batch_masks.append(done)
                self.batch_targets.append(aux['goal_id'])
                self.state = next_state
                self.episode_reward += reward

                if done:
                    if reward > 0: #action == self.env.goal_word:
                        self._winning_steps += self.env.max_turns - self.state.remaining_steps()
                        self._wins += 1
                        self._winning_rewards += self.episode_reward
                    else:
                        self._losses += 1

                    self._total_rewards += self.episode_reward

                    self.done_episodes += 1
                    self.state = self.env.reset()
                    self.total_rewards.append(self.episode_reward)
                    self.episode_reward = 0
                    self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len :]))

            _, last_value = self.forward(self.state.vec)

            returns = self.compute_returns(self.batch_rewards, self.batch_masks, last_value)
            for idx in range(self.hparams.batch_size):
                yield self.batch_states[idx], self.batch_actions[idx], returns[idx], self.batch_targets[idx]

            self.batch_states = []
            self.batch_actions = []
            self.batch_rewards = []
            self.batch_masks = []
            self.batch_targets = []

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
            #advs = (returns - returns.mean()) / (returns.std() + self.eps) - values
            # normalize advantages to train actor
            advs = (advs - advs.mean()) / (advs.std() + self.eps)
            # normalize returns to train critic
            #advs = returns - values
            #advs = returns
            #targets = returns
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
        #total_loss = actor_loss + entropy
        return total_loss

    # def play_game(self):
    #     done = False
    #     batch_states = []
    #     batch_actions = []
    #     batch_rewards = []
    #     batch_masks = []
    #     goal_ids = []
    #     self.episode_reward = 0
    #     with torch.no_grad():
    #         while not done:
    #             action = self.agent(self.state.vec, self.device)[0]
    #             next_state, reward, done, aux = self.env.step(action)
    #
    #             batch_rewards.append(reward)
    #             batch_actions.append(action)
    #             batch_states.append(self.state.vec)
    #             batch_masks.append(done)
    #             goal_ids.append(aux["goal_id"])
    #             self.state = next_state
    #             self.episode_reward += reward
    #
    #         returns = []
    #         g = 0
    #         for r, d in zip(batch_rewards[::-1], batch_masks[::-1]):
    #             g = r + self.hparams.gamma * g * (1 - d)
    #             returns.append(g)
    #
    #         # reverse list and stop the gradients
    #         returns = torch.tensor(returns[::-1])
    #
    #         seq = [
    #             Experience(*x) for x in zip(batch_states, batch_actions, returns, goal_ids)
    #         ]
    #         if batch_rewards[-1] > 0:
    #             self._winning_steps += self.env.max_turns - self.state.remaining_steps()
    #             self._wins += 1
    #             self._winning_rewards += self.episode_reward
    #             self.dataset.winners.append(seq)
    #         else:
    #             self._losses += 1
    #             self.dataset.losers.append(seq)
    #
    #         self._total_rewards += self.episode_reward
    #
    #         self.done_episodes += 1
    #         self.state = self.env.reset()
    #         self.total_rewards.append(self.episode_reward)
    #         self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len:]))

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> OrderedDict:
        """Perform one actor-critic update using a batch of data.
        Args:
            batch: a batch of (states, actions, returns)
        """
        states, actions, returns, goal_ids = batch

        # Play a few games here to generate fresh data
        # with torch.no_grad():
            # for _ in range(self.hparams.batch_size):
            #     self.play_game()

        # Compute loss to backprop
        loss = self.loss(states, actions, returns)

        if self.global_step % 10 == 0:
            # Find a sequence
            i = 0
            while True:
                if states[i][0] == self.env.max_turns:
                    break
                i += 1
            # Walk through game
            game = f"goal: {self.env.words[goal_ids[i]]}\n"
            strt = i
            game += f'{0}: {self.env.words[actions[i]]}\n'
            i += 1
            while states[i][0] != self.env.max_turns:
                game += f'{i-strt}: {self.env.words[actions[i]]}\n'
                i += 1

            self.writer.add_text("game sample", game, global_step=self.global_step)

            # if len(self.dataset.winners) > 0:
                # winner = self.dataset.winners.buffer[-1]
                # game = f"goal: {self.env.words[winner[0].goal_id]}\n"
                # for i, xp in enumerate(winner):
                    # tried = ''.join(
                    #     chr(ord('A') + i)
                    #     for i, seen in enumerate(xp.state[1:27])
                    #     if seen
                    # )
                    # offset = 1+26 + 0
                    # tried = ''.join(
                    #     str(x)
                    #     for x in xp.state[offset:offset+15]
                    # )
                    # game += f"{i}: {self.env.words[xp.action]}\n"
                # self.writer.add_text("game sample/winner", game, global_step=self.global_step)
            # if len(self.dataset.losers) > 0:
            #     loser = self.dataset.losers.buffer[-1]
            #     game = f"goal: {self.env.words[loser[0].goal_id]}\n"
            #     for i, xp in enumerate(loser):
            #         # tried = ''.join(
            #         #     chr(ord('A') + i)
            #         #     for i, seen in enumerate(xp.state[1:27])
            #         #     if seen
            #         # )
            #         # offset = 1+26 + 0
            #         # tried = ''.join(
            #         #     str(x)
            #         #     for x in xp.state[offset:offset+15]
            #         # )
            #         game += f"{i}: {self.env.words[xp.action]}\n"
            #     self.writer.add_text("game sample/loser", game, global_step=self.global_step)
            self.writer.add_scalar("train_loss", loss, global_step=self.global_step)
            self.writer.add_scalar("total_games_played", self.done_episodes, global_step=self.global_step)
            self.writer.add_scalar("winner_buffer", len(self.dataset.winners), global_step=self.global_step)
            self.writer.add_scalar("loser_buffer", len(self.dataset.losers), global_step=self.global_step)

            self.writer.add_scalar("lose_ratio", self._losses/(self._wins+self._losses), global_step=self.global_step)
            self.writer.add_scalar("wins", self._wins, global_step=self.global_step)
            self.writer.add_scalar("reward_per_game", self._total_rewards / (self._wins+self._losses), global_step=self.global_step)
            if self._wins > 0:
                self.writer.add_scalar("reward_per_win", self._winning_rewards / self._wins, global_step=self.global_step)
                self.writer.add_scalar("avg_winning_turns", self._winning_steps/self._wins, global_step=self.global_step)

            self._winning_steps = 0
            self._winning_rewards = 0
            self._total_rewards = 0
            self._wins = 0
            self._losses = 0

        log = {
            "episodes": self.done_episodes,
            "reward": self.total_rewards[-1],
            "avg_reward": self.avg_rewards,
        }
        return OrderedDict(
            {
                "loss": loss,
                "avg_reward": self.avg_rewards,
                "log": log,
                "progress_bar": log,
            }
        )

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        #dataset = ExperienceSourceDataset(self.train_batch)
        # dataloader = DataLoader(dataset=self.dataset, batch_size=self.hparams.batch_size)
        # return dataloader
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
        arg_parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
        arg_parser.add_argument("--env", type=str, default="WordleEnv100-v0", help="gym environment tag")
        arg_parser.add_argument("--network_name", type=str, default="SumChars", help="Network to use")
        arg_parser.add_argument("--hidden_size", type=int, default="256", help="Width of hidden layers")
        arg_parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
        arg_parser.add_argument("--seed", type=int, default=123, help="seed for training run")
        arg_parser.add_argument("--replay_size", type=int, default=1000, help="Size of replay buffer(s)")

        arg_parser.add_argument(
            "--avg_reward_len",
            type=int,
            default=100,
            help="how many episodes to include in avg reward",
        )

        return arg_parser


def cli_main() -> None:
    parser = ArgumentParser(add_help=False)

    # trainer args
    parser = Trainer.add_argparse_args(parser)

    # model args
    parser = AdvantageActorCritic.add_model_specific_args(parser)
    args = parser.parse_args()

    model = AdvantageActorCritic(**args.__dict__)

    # save checkpoints based on avg_reward
    checkpoint_callback = ModelCheckpoint()

    seed_everything(123)
    trainer = Trainer.from_argparse_args(args, deterministic=True, callbacks=checkpoint_callback)
    trainer.fit(model)


if __name__ == '__main__':
    cli_main()
