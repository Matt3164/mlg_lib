from typing import Tuple

from numpy.core._multiarray_umath import ndarray
from sklearn.datasets import fetch_openml

from fmnist_benchmark.common import memory


label_names = ["top", "trouser", "pullover", "dress", "coat",
	"sandal", "shirt", "sneaker", "bag", "ankle boot"]

@memory.cache
def get_fmnist() -> Tuple[ndarray, ndarray]:
    data = fetch_openml("Fashion-MNIST", data_home="/home/matthieu/sklearn_data")
    X = data["data"].reshape(-1, 28, 28)
    Y = data["target"].astype(int)
    return X, Y