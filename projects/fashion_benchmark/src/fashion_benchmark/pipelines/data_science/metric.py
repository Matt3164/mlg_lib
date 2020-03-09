import logging

from typing import Callable, Any

import attr
from numpy.core.multiarray import ndarray

logger = logging.getLogger(__name__)

@attr.s
class Metric(object):
    """"""

    metric_fn = attr.ib(type=Callable[[ndarray, ndarray], Any])

    def __call__(self, Y_true: ndarray, Y_pred: ndarray)->Any:
        metric = self.metric_fn(Y_true, Y_pred)
        logger.info(f"{self.metric_fn.__name__} : {metric}")
        return metric
