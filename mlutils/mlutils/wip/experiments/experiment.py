import logging
import os
from typing import Callable, List

from sklearn.base import BaseEstimator

from mlutils.wip.experiments import exp_utils
from nautilus.context.to_prediction_context import ToPredictionContext
from nautilus.dataset.dataset import Dataset
from mlutils.metrics import BufferMetric
from nautilus.utils import file_utils

logger = logging.getLogger(__name__)


class Experiment(object):
    """"""

    def __init__(self,
                 train_dataset_fn: Callable[[], Dataset],
                 test_dataset_fn: Callable[[], Dataset],
                 model: BaseEstimator,
                 metrics: List[BufferMetric],
                 exp_tag: str,
                 use_cache: bool=True
                 ):
        """Constructor for Experiment"""
        self.train_dataset_fn = train_dataset_fn
        self.test_dataset_fn = test_dataset_fn
        self.model = model
        self.metrics = metrics
        self.exp_tag = exp_tag
        self.use_cache=use_cache

    def root_dir(self):
        return exp_utils.root_dir(self.exp_tag)

    def train_dataset(self):
        return os.path.join(self.root_dir(), "train.npz")

    def test_dataset(self):
        return os.path.join(self.root_dir(), "test.npz")

    def get_train_data(self) -> Dataset:

        if file_utils.exists(self.train_dataset()) and self.use_cache:

            return Dataset.from_file(self.train_dataset())
        else:

            dataset = self.train_dataset_fn()

            if self.use_cache:
                dataset.to_file(self.train_dataset())

            return dataset

    def get_test_data(self) -> Dataset:

        if file_utils.exists(self.test_dataset()) and self.use_cache:

            return Dataset.from_file(self.test_dataset())

        else:

            dataset = self.test_dataset_fn()

            if self.use_cache:
                dataset.to_file(self.test_dataset())

            return dataset

    def run(self):
        exp_utils.init(self.exp_tag)
        train_dataset = self.get_train_data()
        test_dataset = self.get_test_data()
        self.model.fit(train_dataset.x, train_dataset.y.ravel())

        if len(self.metrics)>0:
            self.data = self.get_test_data()
            for fold, dataset in zip(["train", "test"],
                                     [train_dataset, test_dataset]):

                prediction_context = ToPredictionContext(self.model)(dataset)

                for metric in self.metrics:

                    m = metric(prediction_context)

                    logger.info("Fold {0} : metric {1} --> {2}".format(fold,
                                                                       type(
                                                                           metric).__name__,
                                                                       m))

