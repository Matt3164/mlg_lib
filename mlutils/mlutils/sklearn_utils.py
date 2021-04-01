import joblib
from sklearn.base import BaseEstimator


def model_from_path(path: str) -> BaseEstimator:
    return joblib.load(path)


def model_to_path(model: BaseEstimator, path: str) -> str:
    joblib.dump(
        model,
        path
    )
    return path
