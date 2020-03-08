from enum import Enum

from mlg_lib.image.io.cv2_reader import cv2_imread
from mlg_lib.image.io.numpy_reader import numpy_read
from mlg_lib.image.io.skimage_reader import skimage_imread


class ImageReader(Enum):
    numpy = numpy_read
    opencv = cv2_imread
    skimage = skimage_imread


