from dataclasses import dataclass

from numpy.core.multiarray import ndarray

@dataclass
class Sample(object):
    x: ndarray
    y: ndarray

    @staticmethod
    def from_tuple(*args) -> 'Sample':
        return Sample(x=args[0], y=args[1])
