from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Tuple, Any, List

import numpy as np
from torch.utils.data.dataset import IterableDataset


Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state"],
)

# @dataclass
# class Experience:
#     state: Any
#     action: Any
#     reward: Any
#     done: Any
#     new_state: Any
#

class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool),
            np.array(next_states),
        )


class SequenceReplay:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.winners = deque(maxlen=capacity//2)
        self.losers = deque(maxlen=capacity//2)

    # def __len__(self) -> int:
    #     return len(self.winners) + len(self.losers)

    def append_winner(self, xp_seq: List[Experience]) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.winners.append(xp_seq)

    def append_loser(self, xp_seq: List[Experience]) -> None:
        self.losers.append(xp_seq)

    def sample(self, batch_size: int) -> Tuple:
        xps = []
        if len(self.winners) > 0:
            w_indices = np.random.choice(len(self.winners), min(batch_size // 2, len(self.winners)), replace=False)
            for i in w_indices:
                xps.extend(self.winners[i])
                if len(xps) >= batch_size//2:
                    xps = xps[:batch_size//2]
                    break

        l_indices = np.random.choice(len(self.losers), min(batch_size, len(self.losers)), replace=False)
        for i in l_indices:
            xps.extend(self.losers[i])
            if len(xps) >= batch_size:
                xps = xps[:batch_size]
                break

        states, actions, rewards, dones, next_states = zip(*(xp for xp in xps))

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool),
            np.array(next_states),
        )


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: SequenceReplay, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]


