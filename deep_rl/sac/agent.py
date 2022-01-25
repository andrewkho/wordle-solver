from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor, nn


class SoftActorCriticAgent:
    """Actor-Critic based agent that returns a discrete action based on the policy."""
    def __init__(self, net: nn.Module):
        self.net = net

    def __call__(self, states: torch.Tensor, device: str) -> List[int]:
        """Takes in the current state and returns the action based on the agents policy.
        Args:
            states: current state of the environment
            device: the device used for the current batch
        Returns:
            action defined by policy
        """
        logprobs = self.net(torch.tensor([states], device=device))
        probs = logprobs.exp().squeeze(dim=-1)

        return self._sample_actions_from_probs(probs)

    def _sample_actions_from_probs(self, probs: torch.Tensor) -> List[int]:
        prob_np = probs.data.cpu().numpy()

        # take the numpy values and randomly select action based on prob distribution
        #actions = [np.random.choice(len(prob), p=prob) for prob in prob_np]
        cdf = np.cumsum(prob_np, axis=1)
        select = np.random.random(cdf.shape[0])
        actions = []
        for row, srch in enumerate(select):
            a = np.searchsorted(cdf[row, :], srch)
            if a >= cdf.shape[1]:
                print("ERROR!!!", a, cdf.shape[1], srch, cdf[row, :])
                a = cdf.shape[1]-1
            actions.append(a)
        return actions

    def sample_actions_and_logprobs(self, states: Tensor, device: str) -> Tuple[List[int], List[int]]:
        logprobs = self.net(torch.tensor(states, device=device))
        probs = logprobs.exp().squeeze(dim=-1)
        actions = self._sample_actions_from_probs(probs)
        logprobs = logprobs[range(len(actions)), actions]

        return actions, logprobs.view(-1, 1)

    def get_action_greedy(self, states: Tensor, device: str) -> List[int]:
        """Get the action greedily (without sampling)
        Args:
            states: current state of the environment
            device: the device used for the current batch
        Returns:
            action defined by policy
        """
        logprobs = self.net(torch.tensor([states], device=device))
        _, actions = torch.max(logprobs, dim=1)
        return [action.item() for action in actions]
