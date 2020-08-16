from typing import Callable

import numpy
from skimage.feature import local_binary_pattern

def wrapped_lbp(radius: int=3)->Callable[[numpy.ndarray], numpy.ndarray]:
    def feature_fn(x: numpy.ndarray)->numpy.ndarray:
        return local_binary_pattern(x, P=8 * radius, R=radius)
    return feature_fn
