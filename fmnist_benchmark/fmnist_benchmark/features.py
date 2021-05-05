from enum import Enum, auto

import numpy

from iputils.features.convolution import make_convolution_fn
from iputils.features.daisy import make_daisy_from_mode
from iputils.features.flatten import make_raw_features_fn
from iputils.features.haar import make_haar_features_fn, HaarHelpers
from iputils.features.hog import make_hog_fn
from iputils.features.local_features import make_local_features_fn
from iputils.features.unsupervised_kmeans_features import make_unsupervised_kmeans_features
from mlutils.sklearn_utils import model_from_path


class Features(Enum):
    raw = auto()
    daisy = auto()
    hog = auto()
    rocket = auto()
    unsup_km = auto()
    local = auto()
    haar = auto()

    def get_fn(self):
        if self == Features.raw:
            return make_raw_features_fn()
        elif self == Features.daisy:
            return make_daisy_from_mode(mode=0)
        elif self == Features.hog:
            return make_hog_fn()
        elif self == Features.rocket:
            kernel = 0.2 * numpy.random.randn(3, 3, 1024)
            return make_convolution_fn(kernel)
        elif self == Features.unsup_km:
            return make_unsupervised_kmeans_features(model_from_path("/tmp/fmnist_bench/km.pickle"))
        elif self == Features.local:
            return make_local_features_fn()
        elif self == Features.haar:
            return make_haar_features_fn(*HaarHelpers.random_features(28, 28, n_max=1000))
        else:
            raise NotImplementedError
