from dataclasses import dataclass
from typing import List, Union

import numpy

from iputils.data.tensor_list import _TensorList
from iputils.data.tensor_list_factory import TensorList


@dataclass
class LabelList(_TensorList):
    tensor_list: _TensorList
    names: Union[None, List[str]] = None

    def __getitem__(self, item) -> numpy.ndarray:
        return self.tensor_list[item]

    def __len__(self) -> int:
        return len(self.tensor_list)

    def get_name(self, item: int) -> str:
        return self.names[self.tensor_list[item]]

    @staticmethod
    def from_array(
            arr: numpy.ndarray,
            names: List[str] = None
        ) -> 'LabelList':
        return LabelList(tensor_list=TensorList.from_array(arr=arr), names=names)
