import numpy as np
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def predict(model: Pipeline, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


def report_accuracy(predictions: np.ndarray, test_y: np.ndarray) -> None:
    return classification_report(predictions, test_y, output_dict=True)

