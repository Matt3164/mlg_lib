from dataclasses import dataclass
from typing import Union, Tuple, List, Callable

import numpy

from mlg_lib.data.img_label_dataloader import ImgLabelDataloader
from mlg_lib.data.img_lbl_dataset import ImgLabelDataset


@dataclass
class ImgLabelDatabunch(object):
    train_dl : ImgLabelDataloader
    valid_dl: ImgLabelDataloader
    test_dl: Union[ImgLabelDataloader, None]=None

    @staticmethod
    def from_arrays(
            arrays: Tuple[Tuple[numpy.ndarray, numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray]],
            batch_size: int,
            img_transform: Callable[[numpy.ndarray], numpy.ndarray],
            names: List[str]=None,
    )->"ImgLabelDatabunch":
        return ImgLabelDatabunch(
            train_dl=ImgLabelDataloader(dataset=ImgLabelDataset.from_arrays(x=arrays[0][0], y=arrays[0][1], names=names, img_transform=img_transform), batch_size=batch_size),
            valid_dl=ImgLabelDataloader(dataset=ImgLabelDataset.from_arrays(x=arrays[1][0], y=arrays[1][1], names=names,
                                                                            img_transform=img_transform),
                                        batch_size=batch_size)
        )