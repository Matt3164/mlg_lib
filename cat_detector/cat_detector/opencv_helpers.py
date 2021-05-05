from pathlib import Path
from typing import Union

import cv2
from numpy import ndarray


class OpenCVHelpers(object):
    @staticmethod
    def imread(path: Union[Path, str]) -> ndarray:
        return cv2.cvtColor(
            cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)

    @staticmethod
    def imwrite(path: Union[Path, str], arr: ndarray) -> bool:
        return cv2.imwrite(
            str(path),
            cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        )

from_path = OpenCVHelpers.imread
to_path = OpenCVHelpers.imwrite
