import attr
import numpy as np
from skimage.feature import local_binary_pattern


@attr.s
class LBPFactory(object):

    radius=attr.ib(type=int, default=8)

    def __call__(self, arr: np.ndarray)->np.ndarray:
        return local_binary_pattern(arr, 8*self.radius, self.radius, "uniform").flatten()