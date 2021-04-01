import numpy
from skimage.feature import daisy

from iputils.features.computer import FeaturesComputer


def make_daisy_fn(
        step: int = 13,
        radius: int = 7,
        rings: int = 2,
        histograms: int = 8,
        orientations: int = 8) -> FeaturesComputer:
    def feature_fn(x: numpy.ndarray) -> numpy.ndarray:
        return daisy(x,
                     step=step,
                     radius=radius,
                     rings=rings,
                     histograms=histograms,
                     orientations=orientations).flatten()

    return feature_fn


def make_daisy_from_mode(mode: int,
                         rings: int = 2,
                         histograms: int = 8,
                         orientations: int = 8
                         ) -> FeaturesComputer:
    if mode == 0:
        step, radius = 13, 7
    elif mode == 1:
        step, radius = 6, 4
    else:
        raise NotImplementedError

    return make_daisy_fn(step,
                         radius,
                         rings,
                         histograms,
                         orientations)
