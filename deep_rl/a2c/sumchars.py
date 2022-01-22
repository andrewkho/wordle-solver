from typing import List

import numpy as np
import torch
from torch import nn


class SumChars(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, word_list: List[str], hidden_size: int = 256):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        word_width = 26*5
        self.f0 = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, word_width),
            nn.ReLU()
        )
        word_array = np.zeros((word_width, len(word_list)))
        for i, word in enumerate(word_list):
            for j, c in enumerate(word):
                word_array[j*26 + (ord(c) - ord('A')), i] = 1
        self.words = torch.Tensor(word_array)

        self.actor_head = nn.Linear(word_width, word_width)
        self.critic_head = nn.Linear(word_width, 1)

    def forward(self, x):
        x = self.f0(x.float())
        a = torch.log_softmax(
            torch.tensordot(self.actor_head(x),
                            self.words.to(self.get_device(x)),
                            dims=((1,), (0,))),
            dim=1)
        c = self.critic_head(x)
        return a, c

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index