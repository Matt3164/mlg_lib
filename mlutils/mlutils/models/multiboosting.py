import numpy
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, BaggingClassifier


def make_multiboosting(T: int = 100, max_depth: int = 2, max_bins: int = 255) -> BaseEstimator:
    _N = int(numpy.sqrt(T))
    return BaggingClassifier(
        base_estimator=HistGradientBoostingClassifier(max_depth=max_depth, max_iter=_N, max_bins=max_bins),
        max_samples=1. / _N,
        n_estimators=_N
    )
