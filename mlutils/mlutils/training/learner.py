from typing import Callable, Dict

from sklearn.pipeline import Pipeline

from mlutils.data.numpy_dataset import NumpyDatabunch
from mlutils.metrics.buffer_metric import BufferMetric
from mlutils.training.to_prediction_context import to_prediction_ctx_fn
from mlutils.training.training_output import TrainingOutput


def learner(
        model: Pipeline,
        bunch: NumpyDatabunch,
        metrics: Dict[str,BufferMetric]
) -> Callable[[], TrainingOutput]:
    def fit_fn() -> TrainingOutput:
        model.fit(bunch.train.x, bunch.train.y.ravel())

        to_pred_ctx_fn = to_prediction_ctx_fn(model)

        logs = dict()
        for fold, dataset in [("train", bunch.train), ("test", bunch.test)]:
            logs[fold] = dict()
            ctx = to_pred_ctx_fn(dataset)
            for mkey, m in metrics.items():
                logs[fold][mkey] = m(ctx)

        return TrainingOutput(model=model, logs=logs)

    return fit_fn
