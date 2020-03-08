from pathlib import Path
from skimage.io import imread
import numpy as np


def skimage_imread(p: Path)->np.ndarray:
    return imread(str(p))