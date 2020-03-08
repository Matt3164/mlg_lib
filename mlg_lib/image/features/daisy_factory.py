import attr
import numpy as np
from skimage.feature import daisy


@attr.s
class DaisyFactory(object):

    step=attr.ib(type=int, default=8)
    radius=attr.ib(type=int, default=8)
    rings=attr.ib(type=int, default=2)
    histograms=attr.ib(type=int, default=6)
    orientations=attr.ib(type=int, default=8)

    def __call__(self, arr: np.ndarray)->np.ndarray:
        return daisy(arr, step=self.step, radius=self.radius, rings=self.rings, histograms=self.histograms,
                         orientations=self.orientations).flatten()