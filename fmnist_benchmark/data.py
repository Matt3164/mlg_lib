from typing import Tuple

from numpy.core._multiarray_umath import ndarray
from sklearn.datasets import fetch_openml

from common import memory


@memory.cache
def get_fmnist() -> Tuple[ndarray, ndarray]:
    data = fetch_openml("Fashion-MNIST", data_home="/home/matthieu/sklearn_data")
    X = data["data"].reshape(-1, 28, 28)
    Y = data["target"].astype(int)
    return X, Y