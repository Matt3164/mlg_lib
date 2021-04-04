from dataclasses import dataclass
from typing import Tuple, Callable

import numpy

from iputils.data.image_list import _ImageList
from iputils.data.label_list import _LabelList
from mlutils.data.dataset import Dataset
from mlutils.data.sample import Sample


def _identity(arr: numpy.ndarray) -> numpy.ndarray:
    return arr


@dataclass
class ImgLabelDataset(Dataset):
    lbl_list: _LabelList
    img_list: _ImageList
    img_transform: Callable[[numpy.ndarray], numpy.ndarray] = _identity

    def __getitem__(self, item: int) -> Sample:
        return Sample.from_tuple(self.img_transform(self.img_list[item]), self.lbl_list[item])

    def __len__(self) -> int:
        assert len(self.lbl_list) == len(self.img_list)
        return len(self.img_list)
