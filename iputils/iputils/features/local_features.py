import numpy
from skimage.feature import multiscale_basic_features

from iputils.features.computer import FeaturesComputer


def make_local_features_fn(num_sigma: int = 3) -> FeaturesComputer:
    def features_fn(x: numpy.ndarray) -> numpy.ndarray:
        return multiscale_basic_features(x, num_sigma=num_sigma).ravel()

    return features_fn
