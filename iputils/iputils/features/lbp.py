from typing import Callable

import numpy
from skimage.feature import local_binary_pattern

from iputils.features.computer import FeaturesComputer


def make_lbp_fn(n_points: int, radius: int) -> FeaturesComputer:
    def feature_fn(x: numpy.ndarray) -> numpy.ndarray:
        return local_binary_pattern(x, P=n_points, R=radius).ravel()

    return feature_fn


def make_lbp_from_radius(radius: int = 3) -> Callable[[numpy.ndarray], numpy.ndarray]:
    return make_lbp_fn(8 * radius, radius)
