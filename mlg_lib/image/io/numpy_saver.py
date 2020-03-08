from pathlib import Path
import numpy as np

def numpy_save(p: Path, img: np.ndarray):
    np.save(str(p), img)