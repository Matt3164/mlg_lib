from pathlib import Path

from sklearn.datasets import fetch_openml
import numpy as np


if __name__ == '__main__':

    data = fetch_openml("Fashion-MNIST", data_home="/home/matthieu/sklearn_data")

    X = data["data"].reshape(-1, 28, 28)
    Y = data["target"]

    x_file = Path.cwd().parent / "data" / "01_raw" / "fashion_images.npy"
    np.save(str(x_file), X)
    y_file = Path.cwd().parent / "data" / "01_raw" / "fashion_labels.npy"
    np.save(str(y_file), Y)

