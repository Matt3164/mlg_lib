import cv2
import numpy as np
from skimage.feature import multiscale_basic_features

NOMINAL_SIZE = 32


def features_fn(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (NOMINAL_SIZE, NOMINAL_SIZE))
    features = multiscale_basic_features(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    features = features[8:-8, 8:-8, :]
    return features.ravel()
