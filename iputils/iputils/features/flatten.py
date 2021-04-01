import numpy as np

from iputils.features.computer import FeaturesComputer


def make_raw_features_fn() -> FeaturesComputer:
    def flatten(arr: np.ndarray) -> np.ndarray:
        return arr.flatten()

    return flatten
