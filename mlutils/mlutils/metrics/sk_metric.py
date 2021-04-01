from functools import partial
from typing import Callable, Any

from numpy.core._multiarray_umath import ndarray
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from mlutils.metrics.buffer_metric import BufferMetric
from mlutils.training.prediction_context import BufferPredictionCtx


def metric_from_sk_scorer(sk_scorer: Callable[[ndarray, ndarray], Any]) -> BufferMetric:
    def metric_fn(ctx: BufferPredictionCtx) -> Any:
        return sk_scorer(ctx.y_true, ctx.y_pred)

    return metric_fn


def make_accuracy_metric():
    return metric_from_sk_scorer(accuracy_score)


def make_cm_metric():
    return metric_from_sk_scorer(confusion_matrix)


def make_cr_metric():
    return metric_from_sk_scorer(partial(classification_report, output_dict=True))
