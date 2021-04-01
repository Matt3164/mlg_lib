from dataclasses import dataclass
from pathlib import Path

import pandas
from imgaug import BoundingBox
from numpy import ndarray

from pypurr.opencv_helpers import OpenCVHelpers


@dataclass
class ObjectDetectionItem(object):
    bbox: BoundingBox
    img: ndarray


@dataclass
class ObjectDetectionDataset(object):
    """
    Conform to Pytorch Dataset API
    """

    image_df: pandas.DataFrame

    @staticmethod
    def from_path(path: Path) -> 'ObjectDetectionDataset':
        return ObjectDetectionDataset(image_df=pandas.read_csv(path, index_col=0))

    def __getitem__(self, item: int) -> ObjectDetectionItem:
        row = self.image_df.iloc[item]
        return ObjectDetectionItem(
            img=OpenCVHelpers.imread(row["ImagePath"]),
            bbox=BoundingBox(x1=row["Xmin"], x2=row["Xmax"], y1=row["Ymin"], y2=row["Ymax"])
        )

    def __len__(self) -> int:
        return self.image_df.shape[0]
