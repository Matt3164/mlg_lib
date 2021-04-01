import numpy
from skimage.util import view_as_windows
from sklearn.pipeline import Pipeline

from iputils.features.computer import FeaturesComputer


def make_unsupervised_kmeans_features(km: Pipeline) -> FeaturesComputer:
    def features_fn(x: numpy.ndarray) -> numpy.ndarray:
        patches = view_as_windows(x, window_shape=(8, 8), step=8)
        patches = patches.reshape(-1, 8 * 8)
        return km.transform(patches).flatten()

    return features_fn
