from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Callable

import numpy as np

from pypurr.opencv_helpers import OpenCVHelpers

@dataclass
class ImageClassificationItem(object):
    features: np.ndarray
    label: int

@dataclass
class ImageClassificationDataset(object):
    files: List[str]
    labels: List[int]
    features_fn: Callable[[np.ndarray], np.ndarray]

    def __getitem__(self, item: int) -> ImageClassificationItem:
        return ImageClassificationItem(features=self.features_fn(self.get_img(item)), label=self.labels[item])

    def get_img(self, item: int) -> np.ndarray:
        return OpenCVHelpers.imread(self.files[item])

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def from_folders(
            img_dir: Path,
            features_fn: Callable[[np.ndarray], np.ndarray],
            skip_directories: List[str] = None
    ) -> 'ImageClassificationDataset':
        files = list(img_dir.glob("*/*.png"))

        if skip_directories:
            files = list(filter(lambda f: not f.parent.name in skip_directories, files))

        labels = list(map(lambda f: f.parent.name, files))

        return ImageClassificationDataset(
            files=list(map(str, files)),
            labels=labels,
            features_fn=features_fn
        )


def from_path(fn: str) -> Tuple[np.ndarray, np.ndarray]:
    dataset = np.load(fn)
    return dataset["X"], dataset["Y"]


def to_path(fn: str, X: np.ndarray, Y: np.ndarray) -> None:
    np.savez_compressed(
        fn, X=X, Y=Y,
    )

    return None


def from_iterable(
        paths: List[Tuple[np.ndarray, int]],
        target_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.empty((len(paths), target_size, target_size))
    Y = np.empty((len(paths), 1))

    for idx, (img, label) in enumerate(paths):

        X[idx, ::] = img
        Y[idx, 0] = label

    return X, Y
