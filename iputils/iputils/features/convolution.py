from itertools import chain

import cv2
import numpy as np
from toolz import pipe
from toolz.curried import map as map_c

from iputils.features.computer import FeaturesComputer


def make_convolution_fn(kernel: np.ndarray) -> FeaturesComputer:
    def features_fn(arr: np.ndarray) -> np.ndarray:
        return pipe(
            range(kernel.shape[-1]),
            map_c(lambda k: kernel[:, :, k]),
            map_c(lambda k: cv2.filter2D(arr, -1, k)),
            map_c(lambda out: np.concatenate(
                [np.array(cv2.meanStdDev(out)).flatten(), np.array([(out > 0).mean(axis=(0, 1))])], axis=0)),
            chain.from_iterable,
            list,
            np.array
        )

    return features_fn
