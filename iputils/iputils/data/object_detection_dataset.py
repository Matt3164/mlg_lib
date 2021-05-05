from dataclasses import dataclass

import numpy
from matplotlib import pyplot
from skimage.draw import rectangle, rectangle_perimeter

from iputils.data.file_list import FileList
from iputils.data.tensor_list import _TensorList
from mlutils.data.dataset import Dataset
from mlutils.data.sample import Sample


@dataclass
class ObjectDetectionDataset(Dataset):
    bbox_list: FileList
    img_list: _TensorList

    def __getitem__(self, item: int) -> Sample:
        return Sample(x=self.img_list[item], y=self.bbox_list[item])

    def __len__(self) -> int:
        return len(self.img_list)

    def show_batch(self, n: int):
        grid_sz = numpy.floor(numpy.sqrt(n)) + 1

        for _ in range(n):
            pyplot.subplot(grid_sz, grid_sz, _ + 1)

            sample = self[_]
            img = sample.x
            bbox = sample.y
            rr, cc = rectangle_perimeter(start=(bbox[2], bbox[0]), end=(bbox[-1], bbox[1]))
            img[rr, cc] = (0, 255, 0)
            pyplot.imshow(img, cmap="gray")
