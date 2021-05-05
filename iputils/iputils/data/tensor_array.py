from dataclasses import dataclass

import numpy

from iputils.data.tensor_list import _TensorList


@dataclass
class TensorArray(_TensorList):
    arr: numpy.ndarray

    def __getitem__(self, item: int) -> numpy.ndarray:
        if self.arr.ndim > 1:
            return self.arr[item, ::]
        else:
            return self.arr[item]

    def __len__(self) -> int:
        return self.arr.shape[0]
