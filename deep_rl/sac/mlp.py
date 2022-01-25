from typing import Tuple, List

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    """Simple MLP network."""

    def __init__(self,
                 n_input: int,
                 n_output: int,
                 word_list: List[str],
                 hidden_size: int = 128):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        self.word_list = word_list
        word_width = 26*5
        word_array = np.zeros((len(word_list), word_width), dtype=np.int32)
        for i, word in enumerate(word_list):
            for j, c in enumerate(word):
                word_array[i, j*26 + (ord(c) - ord('A'))] = 1
        self.words = torch.Tensor(word_array)
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_output),
        )

    def forward(self, state, action_id):
        """Forward pass through network.
        Args:
            x: input to network
        Returns:
            output of network
        """
        input_x = torch.cat([state, self.words[action_id, :].to(self.get_device(state))], dim=1)
        return self.net(input_x.float())


    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index
