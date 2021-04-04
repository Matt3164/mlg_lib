from dataclasses import dataclass
from typing import List

from mlutils.data.sample import Sample


class Dataset(object):
    def __getitem__(self, item: int) -> Sample:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


@dataclass
class ShuffleDataset(Dataset):
    indexes: List[int]
    dataset: Dataset

    def __getitem__(self, item: int) -> Sample:
        return self.dataset[self.indexes[item]]

    def __len__(self) -> int:
        return len(self.indexes)


