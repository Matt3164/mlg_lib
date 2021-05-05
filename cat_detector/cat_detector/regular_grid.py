from typing import Iterator, Tuple

import numpy
from imgaug import BoundingBox


def regular_grid(shape: Tuple[int, int],
                 size: Tuple[int, int],
                 step: Tuple[int, int]) -> Iterator[BoundingBox]:
    for i in numpy.arange(0, shape[0] - size[0], step[0]):
        for j in numpy.arange(0, shape[1] - size[1], step[1]):
            yield BoundingBox(y1=i, x1=j, x2=j + size[1], y2=i + size[0])
