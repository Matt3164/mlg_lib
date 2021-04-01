from typing import Callable, Union

from sklearn.model_selection import train_test_split

from mlutils.data.numpy_dataset import NumpyDataset, NumpyDatabunch


def make_splitter(
        train_size: Union[float, int, None] = 0.8, test_size: Union[float, int, None] = None, random_state: int = 13) -> \
Callable[[NumpyDataset], NumpyDatabunch]:
    def split_fn(dataset: NumpyDataset) -> NumpyDatabunch:
        xtrain, xtest, ytrain, ytest = train_test_split(dataset.x, dataset.y,
                                                        train_size=train_size,
                                                        test_size=test_size,
                                                        random_state=random_state)

        return NumpyDatabunch(
            train=NumpyDataset.from_arrays(xtrain, ytrain),
            test=NumpyDataset.from_arrays(xtest, ytest)
        )

    return split_fn

# def to_iterator(dataset: Dataset) -> Iterator[Tuple[ndarray, ndarray]]:
#     for i in range(dataset.x.shape[0]):
#         yield dataset.x[i, ::], dataset.y[i, :]
#
#
# def from_iterator(iterator: Iterator[Tuple[ndarray, ndarray]]) -> Dataset:
#     l = list(iterator)
#
#     return Dataset.from_arrays(
#         X=np_utils.nparray(list(map(lambda x: x[0], l))),
#         Y=np_utils.nparray(list(map(lambda x: x[1], l)))
#     )
#
#
# def map_x(dataset: Dataset, map_fn: Callable[[ndarray], ndarray]) -> Dataset:
#     return Dataset.from_arrays(
#         np_utils.map_arr(dataset.x, map_fn),
#         dataset.y
#     )

# def flatmap(dataset: Dataset, arr_func: Callable[[Tuple[ndarray, ndarray]],
#                                                  Iterator[Tuple[ndarray,
#                                                                 ndarray]]]):
#     xy_it = to_iterator(dataset)
#
#     xy_iter = chain.from_iterable(map(lambda xy: arr_func(xy), xy_it))
#
#     return from_iterator(xy_iter)

#
# def on_x(dataset: Dataset, trs: Callable[[ndarray], ndarray])->Dataset:
#     return Dataset.from_arrays(
#         X=trs(dataset.x),
#         Y=dataset.y,
#     )
