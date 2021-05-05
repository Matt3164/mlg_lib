import numpy


class _TensorList(object):
    def __getitem__(self, item) -> numpy.ndarray:
        ...

    def __len__(self) -> int:
        ...
