import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from typing import Callable

from mlg_lib.ml.lambda_row import LambdaRow


def make_single_feature_model(
    feature_fn: Callable[[np.ndarray], np.ndarray],
    clf: Pipeline
    )->Pipeline:

    return make_pipeline(
        LambdaRow(feature_fn),
        clf
    )