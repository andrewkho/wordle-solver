from typing import List

from torch import nn


class MLP(nn.Module):
    """Simple MLP network."""

    def __init__(self, obs_size: int, n_actions: int, word_list: List[str], hidden_size: int = 256):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        self.f0 = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
            nn.Softmax(),
        )

    def forward(self, x):
        return self.f0(x.float())

