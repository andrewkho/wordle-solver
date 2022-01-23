from typing import List

import numpy as np
import torch
from torch import nn


class EmbeddingChars(nn.Module):
    def __init__(self, obs_size: int,
                 word_list: List[str],
                 n_hidden: int = 0,
                 hidden_size: int = 256,
                 embedding_dim: int=4):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        word_width = 26*5
        self.embedding_layer = nn.Embedding(num_embeddings=obs_size,
                                            embedding_dim=embedding_dim)
        layers = [
            nn.Linear(obs_size*embedding_dim, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(n_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, word_width))
        layers.append(nn.ReLU())

        self.f0 = nn.Sequential(*layers)
        word_array = np.zeros((word_width, len(word_list)))
        for i, word in enumerate(word_list):
            for j, c in enumerate(word):
                word_array[j*26 + (ord(c) - ord('A')), i] = 1
        self.words = torch.Tensor(word_array)

        self.actor_head = nn.Linear(word_width, word_width)
        self.critic_head = nn.Linear(word_width, 1)

    def forward(self, x):
        emb = self.embedding_layer(x.int())
        y = self.f0(emb.view(x.shape[0], x.shape[1]*self.embedding_layer.embedding_dim))
        a = torch.log_softmax(
            torch.tensordot(self.actor_head(y),
                            self.words.to(self.get_device(y)),
                            dims=((1,), (0,))),
            dim=-1)
        c = self.critic_head(y)
        return a, c

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index