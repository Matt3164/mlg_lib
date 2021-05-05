from pathlib import Path

from matplotlib import pyplot

from cat_detector.dataset import bbox_reader
from iputils.data.file_list import FileList
from iputils.data.image_list import ImageList
from iputils.data.object_detection_dataset import ObjectDetectionDataset

if __name__ == '__main__':
    img_list = ImageList.from_path(Path("/DATADRIVE1/mlegoff/workspace/data/cats/raw"), "*/*.jpg")
    dataset = ObjectDetectionDataset(
        img_list=img_list,
        bbox_list=FileList.with_suffix(img_list, ".jpg.cat", reader=bbox_reader)
    )

    dataset.show_batch(16)
    pyplot.show()
