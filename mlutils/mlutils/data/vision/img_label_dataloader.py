from dataclasses import dataclass
from typing import Tuple

import numpy

from mlg_lib_deprecated.mlg_lib import ImgLabelDataset


@dataclass
class ImgLabelDataloader(object):
    dataset : ImgLabelDataset
    batch_size: int

    sampled_indexes: numpy.ndarray=None
    idx: int = 0

    def __len__(self) -> int:
        return int(len(self.dataset)/self.batch_size)

    def __iter__(self)->"ImgLabelDataLoader":
        self.idx = 0
        self.sampled_indexes = numpy.random.permutation(numpy.arange(len(self.dataset)))
        return self

    def __next__(self)->Tuple[numpy.ndarray, numpy.ndarray]:
        if self.idx<len(self.dataset):
            data = list([ self.dataset[self.sampled_indexes[_]]  for _ in range(self.idx, min(len(self.dataset), self.idx+self.batch_size))])
            self.idx+=self.batch_size
            return numpy.asarray(list(map(lambda _: _[0], data))),\
                   numpy.asarray(list(map(lambda _: _[1], data)))
        else:
            raise StopIteration


