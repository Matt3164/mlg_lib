from typing import Callable

import numpy
from skimage.feature import hog

def wrapped_hog(
        orientations: int=9,
        transform_sqrt: bool=False,
        pixels_per_cell : int = 8,
        block_norm: str='L2-Hys')->Callable[[numpy.ndarray], numpy.ndarray]:

    def feature_fn(x: numpy.ndarray)->numpy.ndarray:
        return hog(x,
                   orientations=orientations,
                   pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                   cells_per_block=(3, 3),
                   block_norm=block_norm,
                   transform_sqrt=transform_sqrt,
                   feature_vector=True)
    return feature_fn



