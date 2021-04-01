from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from numpy.core.multiarray import ndarray

from mlutils import np_utils
import numpy as np

@dataclass
class NumpyDataset(object):
    x: ndarray
    y: ndarray

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item) -> Tuple[ndarray, ndarray]:
        return self.x[item, ::], self.y[item, ::]

    def __repr__(self):
        return f"NumpyDataset (x: {self.x.shape}, y: {self.y.shape})"

    @staticmethod
    def from_arrays(X: ndarray, Y: ndarray):
        return NumpyDataset(x=X, y=Y)

    @staticmethod
    def from_file(filepath: str):
        data = np.load(filepath, allow_pickle=True)

        return NumpyDataset(
            x=data["X"],
            y=data["Y"],
        )

    def to_file(self, filepath: str):
        np_utils.save_npz(filepath, dict(X=self.x, Y=self.y))

        return Path(filepath).exists()


@dataclass
class NumpyDatabunch:
    train: NumpyDataset = None
    test: NumpyDataset = None
    valid: NumpyDataset = None
