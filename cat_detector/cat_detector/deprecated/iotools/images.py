from os.path import join, exists
from typing import Iterator, Tuple
from numpy.core.multiarray import ndarray

import pypurr.opencv_interface
from pypurr.deprecated.iotools import image
from pypurr.deprecated.iotools.path import find_pngs

def from_path(path: str)->Iterator[Tuple[str, ndarray]]:
    return map(
        lambda x: (x, pypurr.opencv_interface.from_path(x)),
        find_pngs(path)
    )

def to_path(path: str, images: Iterator[ndarray])->bool:

    for idx, img in enumerate(images):

        fn = join(path, "im_{:05d}.png".format(idx))

        if not exists(fn):
            pypurr.opencv_interface.to_path(fn, img)

    return True