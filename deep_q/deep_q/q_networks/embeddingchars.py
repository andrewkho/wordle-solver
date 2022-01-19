from typing import List

import numpy as np
import torch
from torch import nn


class EmbeddingChars(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, word_list: List[str], hidden_size: int = 256):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        word_width = 26*5
        emb_size = 8
        self.embedding_layer = nn.Embedding(obs_size, emb_size)
        self.f0 = nn.Sequential(
            nn.Linear(obs_size*emb_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, word_width),
        )
        word_array = np.zeros((word_width, len(word_list)))
        for i, word in enumerate(word_list):
            for j, c in enumerate(word):
                word_array[j*26 + (ord(c) - ord('A')), i] = 1
        self.words = torch.Tensor(word_array)

    def forward(self, x):
        emb = self.embedding_layer(x.int())
        y = self.f0(emb.view(x.shape[0], x.shape[1]*self.embedding_layer.embedding_dim))
        z = torch.tensordot(y, self.words.to(self.get_device(x)), dims=[(1,), (0,)])
        return nn.Softmax(dim=1)(z)

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index
