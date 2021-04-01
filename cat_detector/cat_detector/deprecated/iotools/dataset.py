from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pypurr.train.helpers.dataset.classif import ImageClassificationDataset


@dataclass
class SkDataset:
    X: np.ndarray
    Y: np.ndarray

    @staticmethod
    def from_path(fn: Path) -> 'SkDataset':
        dataset = np.load(str(fn), allow_pickle=True)
        return SkDataset(X=dataset["X"], Y=dataset["Y"])

    def to_path(self, fn: Path) -> None:
        np.savez_compressed(
            str(fn), X=self.X, Y=self.Y,
        )

        return None


class ImageClassificationDatasetHelpers:
    @staticmethod
    def to_sk_dataset(
            dataset: ImageClassificationDataset) -> SkDataset:

        size = dataset[0].features.shape[0]

        n_samples = len(dataset)
        X = np.empty((n_samples, size))
        Y = np.empty((n_samples, 1), dtype=np.object)

        for idx in range(n_samples):
            item = dataset[idx]
            X[idx, ::] = item.features
            Y[idx, 0] = item.label

        return SkDataset(X=X, Y=Y)
