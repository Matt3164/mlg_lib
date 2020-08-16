from typing import Callable

import numpy
import sklearn
from sklearn.pipeline import Pipeline


def ensemble_update(
    estimators_by_step: int
    )->Callable[[Pipeline, numpy.ndarray, numpy.ndarray], Pipeline]:
    def update_fn(ensemble: Pipeline, x: numpy.ndarray, y: numpy.ndarray)->Pipeline:
        ensemble.n_estimators+=estimators_by_step
        ensemble.fit(x, y)
        return ensemble
    return update_fn

def hist_gbt_update(
    iterations_by_step: int
    )->Callable[[Pipeline, numpy.ndarray, numpy.ndarray], Pipeline]:
    def update_fn(ensemble: Pipeline, x: numpy.ndarray, y: numpy.ndarray)->Pipeline:
        ensemble.max_iter += iterations_by_step
        ensemble.fit(x, y)
        return ensemble
    return update_fn