from sklearn.base import BaseEstimator

from mlutils.data.numpy_dataset import NumpyDataset
from mlutils.training.prediction_context import BufferPredictionCtx


def to_prediction_ctx_fn(model: BaseEstimator):
    def fn(dataset: NumpyDataset) -> BufferPredictionCtx:
        Ytrue = dataset.y.reshape(-1, 1)

        Ypred = model.predict_proba(dataset.x)

        return BufferPredictionCtx(
            y_true=Ytrue,
            y_pred=Ypred.argmax(axis=1),
            y_pred_prob=Ypred
        )

    return fn
