import cv2
import numpy as np

from pypurr.common.config import NOMINAL_SIZE


def resize_to_gray_flatten(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (NOMINAL_SIZE, NOMINAL_SIZE))
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).flatten()
