from pathlib import Path

import imageio
import numpy
from matplotlib import pyplot

from iputils.data.file_list import FileList
from iputils.data.tensor_array import TensorArray
from iputils.data.tensor_list import _TensorList


class ImageList(_TensorList):

    @staticmethod
    def imread(path: Path) -> numpy.ndarray:
        return imageio.imread(str(path))

    @staticmethod
    def from_array(arr: numpy.ndarray) -> _TensorList:
        return TensorArray(arr=arr)

    @staticmethod
    def from_path(folder: Path, glob: str) -> "FileList":
        return FileList(files=list(folder.glob(glob)), reader=ImageList.imread)

    @staticmethod
    def show_batch(img_list: _TensorList, n: int):
        grid_sz = numpy.floor(numpy.sqrt(n)) + 1

        for _ in range(n):
            pyplot.subplot(grid_sz, grid_sz, _ + 1)
            pyplot.imshow(img_list[_], cmap="gray")
