from typing import Union, Callable

import numpy
from sklearn.model_selection import train_test_split

from mlutils.data.databunch import Databunch
from mlutils.data.dataloader import Dataloader
from mlutils.data.dataset import Dataset, ShuffleDataset


class DatasetHelpers:
    @staticmethod
    def shuffle(dataset: Dataset) -> ShuffleDataset:
        return ShuffleDataset(dataset=dataset, indexes=numpy.random.permutation(numpy.arange(len(dataset))).tolist())

    @staticmethod
    def splitter(
            train_size: Union[float, int, None] = 0.8, test_size: Union[float, int, None] = None,
            random_state: int = 13) -> \
            Callable[[Dataset], Databunch]:
        def split_fn(dataset: Dataset) -> Databunch:
            itrain, itest = train_test_split(numpy.arange(len(dataset)),
                                             train_size=train_size,
                                             test_size=test_size,
                                             random_state=random_state)

            return Databunch(
                train_dl=Dataloader(ShuffleDataset(dataset=dataset, indexes=itrain)),
                test_dl=Dataloader(ShuffleDataset(dataset=dataset, indexes=itest))
            )

        return split_fn
