from typing import List

import numpy as np
import torch
from torch import nn


class EmbeddingChars(nn.Module):
    def __init__(self,
                 obs_size: int,
                 word_list: List[str],
                 n_hidden: int = 1,
                 hidden_size: int = 256,
                 n_emb: int = 32,
                 ):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        word_width = 26*5
        self.n_emb = n_emb

        layers = [
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(n_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, self.n_emb))

        self.f_state = nn.Sequential(*layers)

        self.actor_head = nn.Linear(self.n_emb, self.n_emb)
        self.critic_head = nn.Linear(self.n_emb, 1)

        word_array = np.zeros((len(word_list), word_width))
        for i, word in enumerate(word_list):
            for j, c in enumerate(word):
                word_array[i, j*26 + (ord(c) - ord('A'))] = 1
        self.words = torch.Tensor(word_array)

        # W x word_width -> W x emb
        self.f_word = nn.Sequential(
            nn.Linear(word_width, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_emb),
        )

    def forward(self, x):
        fs = self.f_state(x.float())
        fw = self.f_word(
            self.words.to(self.get_device(x)),
        ).transpose(0, 1)

        a = torch.log_softmax(
            torch.tensordot(self.actor_head(fs), fw,
                            dims=((1,), (0,))),
            dim=-1)
        c = self.critic_head(fs)
        return a, c

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index