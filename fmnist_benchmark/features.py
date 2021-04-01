from enum import Enum, auto

import numpy
from numpy.core._multiarray_umath import ndarray

from common import memory
from iputils.features.convolution import make_convolution_fn
from iputils.features.daisy import make_daisy_from_mode
from iputils.features.flatten import make_raw_features_fn
from iputils.features.hog import make_hog_fn
from iputils.features.unsupervised_kmeans_features import make_unsupervised_kmeans_features
from mlutils.np_utils import map_arr
from mlutils.sklearn_utils import model_from_path

class Features(Enum):
    raw = auto()
    daisy = auto()
    hog = auto()
    rocket = auto()
    unsup_km = auto()

    def get_fn(self):
        if self == Features.raw:
            return make_raw_features_fn()
        elif self == Features.daisy:
            return make_daisy_from_mode(mode=0)
        elif self == Features.hog:
            return make_hog_fn()
        elif self == Features.rocket:
            kernel = 0.2 * numpy.random.randn(3, 3, 256)
            return make_convolution_fn(kernel)
        elif self == Features.unsup_km:
            return make_unsupervised_kmeans_features(model_from_path("/tmp/fmnist_bench/km.pickle"))
        else:
            raise NotImplementedError


@memory.cache
def compute_features(X: ndarray, feat: Features) -> ndarray:
    features_fn = feat.get_fn()
    return map_arr(X, features_fn)
