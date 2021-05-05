from dataclasses import dataclass

import numpy
from tqdm import tqdm

from mlutils.data.dataset import Dataset
from mlutils.data.numpy_dataset import NumpyDataset


@dataclass
class Dataloader(object):
    dataset: Dataset
    batch_size: int = -1

    def __post_init__(self):
        if self.batch_size == -1:
            self.batch_size = len(self.dataset)

    def __len__(self) -> int:
        return int(len(self.dataset) / self.batch_size)

    def __iter__(self) -> "ImgLabelDataLoader":
        self.idx = 0
        return self

    def __next__(self) -> NumpyDataset:
        if self.idx < len(self.dataset):
            samples = list([self.dataset[_] for _ in
                            range(self.idx, min(len(self.dataset), self.idx + self.batch_size))])
            self.idx += self.batch_size
            return NumpyDataset(
                x=numpy.asarray(list(map(lambda _: _.x, samples))),
                y=numpy.asarray(list(map(lambda _: _.y, samples))))
        else:
            raise StopIteration

    def stack_one_epoch(self) -> NumpyDataset:
        samples = list([self.dataset[_] for _ in
                        tqdm(range(len(self.dataset)))])
        return NumpyDataset(
            x=numpy.asarray(list(map(lambda _: _.x, samples))),
            y=numpy.asarray(list(map(lambda _: _.y, samples))))
