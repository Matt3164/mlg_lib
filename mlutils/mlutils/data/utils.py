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
