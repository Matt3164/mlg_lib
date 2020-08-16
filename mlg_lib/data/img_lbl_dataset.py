from dataclasses import dataclass
from typing import Tuple, List, Union, Callable

import numpy
from matplotlib import pyplot
from toolz import identity

from mlg_lib.data.dataset import Dataset
from mlg_lib.data.image_list import ImageList, _ImageList
from mlg_lib.data.label_list import LabelList, _LabelList


@dataclass
class ImgLabelDataset(Dataset):
    lbl_list: _LabelList
    img_list: _ImageList
    img_transform : Callable[[numpy.ndarray], numpy.ndarray]= identity

    def __getitem__(self, item: int)->Tuple[numpy.ndarray, int]:
        return self.img_transform(self.img_list[item]), self.lbl_list[item]

    def __len__(self) -> int:
        assert len(self.lbl_list)==len(self.img_list)
        return len(self.img_list)

    def show_batch(self, n: int):
        grid_sz = numpy.floor(numpy.sqrt(n)) + 1

        for _ in range(n):
            pyplot.subplot(grid_sz, grid_sz, _ + 1)

            if self.lbl_list.names is None:
                lbl = self.lbl_list[_]
            else:
                lbl = self.lbl_list.names[self.lbl_list[_]]

            pyplot.title(f"Class : {lbl}")
            pyplot.imshow(self.img_list[_], cmap="gray")


    @staticmethod
    def from_arrays(
            x: numpy.ndarray,
            y: numpy.ndarray,
            names: Union[List[str]]=None,
            img_transform : Callable[[numpy.ndarray], numpy.ndarray]=identity
    )->"ImgLabelDataset":
        return ImgLabelDataset(
            img_list=ImageList.from_array(x),
            lbl_list=LabelList.from_array(y, names=names),
            img_transform=img_transform
        )