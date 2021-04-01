from typing import Callable, Any

from mlutils.training.prediction_context import BufferPredictionCtx

BufferMetric = Callable[[BufferPredictionCtx], Any]
