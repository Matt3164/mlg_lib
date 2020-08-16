from dataclasses import dataclass
from typing import List, Union

import numpy


@dataclass
class _LabelList(object):

    names : Union[None, List[str]]=None

    def __getitem__(self, item: int)->int:
        raise NotImplementedError

    def __len__(self)->int:
        raise NotImplementedError


@dataclass
class LabelArray(_LabelList):

    arr : numpy.ndarray = numpy.empty((0,))

    def __getitem__(self, item: int) -> numpy.ndarray:
        return self.arr[item]

    def __len__(self) -> int:
        return self.arr.shape[0]

class LabelList(_LabelList):
    @staticmethod
    def from_array(
            arr: numpy.ndarray,
            names: List[str]=None
    )->LabelArray:
        return LabelArray(arr=arr, names=names)
