from collections import namedtuple
from typing import Iterator, Callable

from torch.utils.data.dataset import IterableDataset

Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "returns", "goal_id"],
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


