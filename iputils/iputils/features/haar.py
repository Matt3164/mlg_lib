from typing import List

import numpy
from skimage.data import chelsea
from skimage.feature import haar_like_feature, haar_like_feature_coord
from skimage.transform import integral_image

from iputils.features.computer import FeaturesComputer


def make_haar_features_fn(coords: List[numpy.ndarray], types: List[str]) -> FeaturesComputer:
    def features_fn(x: numpy.ndarray) -> numpy.ndarray:
        ii = integral_image(x)
        return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                                 feature_type=types,
                                 feature_coord=coords
                                 )

    return features_fn


class HaarHelpers:
    features = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4']

    @staticmethod
    def random_features(width: int, height: int, n_max: int = 100):
        _coord, _type = haar_like_feature_coord(width=width, height=height, feature_type=HaarHelpers.features)

        idx = numpy.random.permutation(_coord.shape[0])[:n_max]

        return _coord[idx], _type[idx]


if __name__ == '__main__':
    im = chelsea()[:, :, 0]

    _sub_coord, _sub_type = HaarHelpers.random_features(width=28, height=28)

    print(make_haar_features_fn(_sub_coord, _sub_type)(im).shape)
