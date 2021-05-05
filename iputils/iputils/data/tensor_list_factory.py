import numpy

from iputils.data.tensor_array import TensorArray
from iputils.data.tensor_list import _TensorList


class TensorList(object):
    @staticmethod
    def from_array(arr: numpy.ndarray) -> _TensorList:
        return TensorArray(arr=arr)
