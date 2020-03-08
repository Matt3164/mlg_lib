from pathlib import Path
import numpy as np

def numpy_read(p: Path)-> np.ndarray:
    return np.load(str(p))