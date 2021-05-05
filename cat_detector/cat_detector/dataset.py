from pathlib import Path

import numpy

from iputils.data.file_list import FileList
from iputils.data.image_list import ImageList
from iputils.data.object_detection_dataset import ObjectDetectionDataset


class DatasetHelpers:
    @staticmethod
    def from_path(path: Path) -> ObjectDetectionDataset:
        img_list = ImageList.from_path(path, "*/*.jpg")

        return ObjectDetectionDataset(
            img_list=img_list,
            bbox_list=FileList.with_suffix(img_list, ".jpg.cat", reader=DatasetHelpers.bbox_reader)
        )

    @staticmethod
    def bbox_reader(file: Path) -> numpy.ndarray:
        with file.open("r") as f:
            content = f.read()
        pos = [int(e) for e in content.split(" ") if e != '']

        pos = numpy.array(pos)[1:]

        x = pos[::2]

        y = pos[1::2]

        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

        return numpy.array([xmin, xmax, ymin, ymax])
