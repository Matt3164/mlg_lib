from itertools import chain

import attr
import cv2
import numpy as np
from toolz import pipe
from toolz.curried import map as map_c


@attr.s
class ConvolutionFeature(object):
    kernel = attr.ib(type=np.ndarray)
    def __call__(self, arr: np.ndarray)->np.ndarray:
        return pipe(
                range(self.kernel.shape[-1]),
                map_c(lambda k: self.kernel[:,:,k]),
                map_c(lambda k: cv2.filter2D(arr, -1, k)),
                map_c(lambda out: np.concatenate([ np.array(cv2.meanStdDev(out)).flatten() , np.array([(out>0).mean(axis=(0,1))]) ], axis=0)),
                chain.from_iterable,
                list,
                np.array
            )


def convolution(k: np.ndarray)->ConvolutionFeature:
    return ConvolutionFeature(kernel=k)