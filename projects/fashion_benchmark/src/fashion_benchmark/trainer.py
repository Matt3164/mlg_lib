import attr
from sklearn.pipeline import Pipeline
import numpy as np

@attr.s
class Trainer(object):
    """"""

    estimator = attr.ib(type=Pipeline)

    def __call__(self, X: np.ndarray, Y: np.ndarray)->Pipeline:

        self.estimator.fit(X, Y)

        return self.estimator
