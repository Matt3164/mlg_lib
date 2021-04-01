from typing import Tuple

import numpy as np


def _crop(arr: np.ndarray, bb: Tuple[int, int, int, int]) -> np.ndarray:
    return crop(arr, (bb[0], bb[1]), (bb[2], bb[3]))


def crop(arr: np.ndarray, top_left: Tuple[int, int], size: Tuple[int, int]) -> np.ndarray:
    return arr[top_left[0]:(top_left[0] + size[0]), top_left[1]:(top_left[1] + size[1]), :]
