from typing import Tuple, List

import numpy as np
import torch
from torch import nn
import gym

from wordle import wordle
from deep_q.experience import SequenceReplay, Experience


class ActorCriticAgent:
    """Actor-Critic based agent that returns an action based on the networks policy."""

    def __init__(self, net):
        self.net = net

    def __call__(self, states: torch.Tensor, device: str) -> List[int]:
        """Takes in the current state and returns the action based on the agents policy.
        Args:
            states: current state of the environment
            device: the device used for the current batch
        Returns:
            action defined by policy
        """
        if not isinstance(states, list):
            states = [states]

        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, device=device)

        logprobs, _ = self.net(states)
        probabilities = logprobs.exp().squeeze(dim=-1)
        prob_np = probabilities.data.cpu().numpy()

        # take the numpy values and randomly select action based on prob distribution
        actions = [np.random.choice(len(prob), p=prob) for prob in prob_np]

        return actions


class Agent:
    """Base Agent class handeling the interaction with the environment."""

    def __init__(self,
                 env: gym.Env,
                 winner_buffer: SequenceReplay,
                 loser_buffer: SequenceReplay,
                 ) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.winner_buffer = winner_buffer
        self.loser_buffer = loser_buffer
        self.state: wordle.WordleState = self.env.reset()

    def reset(self) -> None:
        """Resents the environment and updates the state."""
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor([self.state.vec]).to(device)
            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    def play_game(
            self,
            net: nn.Module,
            epsilon: float = 0.0,
            device: str = "cpu",
    ) -> Tuple[float, bool]:

        done = False
        cur_seq = list()
        reward = 0
        while not done:
            reward, done, exp = self.play_step(net, epsilon, device)
            cur_seq.append(exp)

        winning_steps = self.env.max_turns - self.state.remaining_steps()
        if reward > 0:
            self.winner_buffer.append(cur_seq)
        else:
            self.loser_buffer.append(cur_seq)
        self.reset()

        return reward, winning_steps

    @torch.no_grad()
    def play_step(
            self,
            net: nn.Module,
            epsilon: float = 0.0,
            device: str = "cpu",
    ) -> Tuple[float, bool, Experience]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """

        action = self.get_action(net, epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)

        exp = Experience(self.state.vec, action, reward, done, new_state.vec)

        self.state = new_state
        return reward, done, exp
