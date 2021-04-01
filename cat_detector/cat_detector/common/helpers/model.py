from pathlib import Path
from typing import Union

import joblib
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class SklearnHelpers:
    @staticmethod
    def clf_from_path(fn: Union[str, Path]) -> Pipeline:
        return joblib.load(fn)

    @staticmethod
    def to_path(fn: Union[str, Path], clf: BaseEstimator) -> None:
        joblib.dump(clf, fn)
        return None
