import numpy as np
from skimage.feature import hog
from typing import Tuple

import attr


@attr.s
class HogFactory(object):

    orientations=attr.ib(type=int, default=8)
    pixels_per_cell=attr.ib(type=Tuple[int,int], default=(8, 8))
    cells_per_block=attr.ib(type=Tuple[int,int], default=(1, 1))
    multichannel=attr.ib(type=bool, default=False)

    def __call__(self, arr: np.ndarray)->np.ndarray:
        return hog(arr,
               orientations=self.orientations,
               pixels_per_cell=self.pixels_per_cell,
               cells_per_block=self.cells_per_block,
               block_norm="L2",
               transform_sqrt=True,
               feature_vector=True,
               multichannel=self.multichannel
                  )