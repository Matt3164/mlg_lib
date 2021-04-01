from dataclasses import dataclass

import numpy
from matplotlib import pyplot


class _ImageList(object):
    def __getitem__(self, item: int)->numpy.ndarray:
        raise NotImplementedError

    def __len__(self)->int:
        raise NotImplementedError

    def show_batch(self, n: int):
        grid_sz = numpy.floor(numpy.sqrt(n)) + 1

        for _ in range(n):
            pyplot.subplot(grid_sz, grid_sz, _+1)
            pyplot.imshow(self[_], cmap="gray")

@dataclass
class ImageArray(_ImageList):
    arr : numpy.ndarray

    def __getitem__(self, item: int) -> numpy.ndarray:
        return self.arr[item, ::]

    def __len__(self) -> int:
        return self.arr.shape[0]

def _img_list_from_arr(arr: numpy.ndarray)->ImageArray:
    return ImageArray(arr=arr)

class ImageList(_ImageList):
    @staticmethod
    def from_array(arr: numpy.ndarray)->ImageArray:
        return ImageArray(arr=arr)

