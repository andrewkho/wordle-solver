import pickle
from collections import deque, namedtuple
from typing import Tuple, List, Iterator, Callable

import numpy as np
from torch.utils.data.dataset import IterableDataset


Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "returns"],
)


class ExperienceSourceDataset(IterableDataset):
    """Basic experience source dataset.
    Takes a generate_batch function that returns an iterator. The logic for the experience source and how the batch is
    generated is defined the Lightning model itself
    """

    def __init__(self, generate_batch: Callable) -> None:
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterator:
        iterator = self.generate_batch()
        return iterator


class SequenceReplay:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int, initialize_winning_replays: str=None) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

        if initialize_winning_replays:
            with open(initialize_winning_replays, 'rb') as f:
                init = pickle.load(f)
            self.buffer.extend(init)

    def __len__(self) -> int:
        return len(self.buffer)

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self.buffer, f)

    def append(self, xp_seq: List[Experience]) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(xp_seq)

    def sample(self, batch_size: int) -> List[Experience]:
        xps = []
        if len(self.buffer) > 0:
            indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
            for i in indices:
                xps.extend(self.buffer[i])
                if len(xps) >= batch_size:
                    xps = xps[:batch_size]
                    break
        return xps


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self,
                 winners: SequenceReplay,
                 losers: SequenceReplay,
                 sample_size: int = 200) -> None:
        self.winners = winners
        self.losers = losers
        self.sample_size = sample_size
        assert self.sample_size % 2 == 0

    def __iter__(self) -> Tuple:
        xps = self.winners.sample(self.sample_size//2) + self.losers.sample(self.sample_size//2)

        states, actions, returns = zip(*xps)
        returns = np.array(returns, dtype=np.float32)

        for i in range(len(actions)):
            yield states[i], actions[i], returns[i]


