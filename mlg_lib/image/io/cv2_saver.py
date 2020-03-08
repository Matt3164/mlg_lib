import cv2
from pathlib import Path

import numpy as np

def cv2_imwrite(p: Path, img: np.ndarray):
    cv2.imwrite(str(p), img)