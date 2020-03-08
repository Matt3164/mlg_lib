from pathlib import Path
import numpy as np
import cv2


def cv2_imread(p: Path)->np.ndarray:
    return cv2.imread(str(p))