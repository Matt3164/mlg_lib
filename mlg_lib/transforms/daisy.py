from typing import Callable

import numpy
from skimage.feature import daisy


def wrapped_daisy(
            step: int=13,
            radius: int=7,
            rings: int=2,
            histograms: int=8,
            orientations: int=8)->Callable[[numpy.ndarray], numpy.ndarray]:

    # if mode==1:
    #     step, radius=13, 7
    # elif mode==2:
    #     step, radius=6, 4
    # else:
    #     step, radius=13, 7

    def feature_fn(x: numpy.ndarray)->numpy.ndarray:
        return daisy(x,
                     step=step,
                     radius=radius,
                     rings=rings,
                     histograms=histograms,
                     orientations=orientations).flatten()
    return feature_fn