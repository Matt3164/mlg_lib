from enum import Enum

from mlg_lib.image.io.cv2_saver import cv2_imwrite
from mlg_lib.image.io.numpy_saver import numpy_save
from mlg_lib.image.io.skimage_saver import skimage_imsave


class ImageSaver(Enum):
    """"""

    opencv = cv2_imwrite
    skimage = skimage_imsave
    numpy = numpy_save
