from itertools import chain
from typing import List, Callable, Tuple

from imgaug import BoundingBox

from cat_detector.regular_grid import regular_grid

WindowExtractor = Callable[[Tuple[int, int]], List[BoundingBox]]


class SlidingWindow:
    @staticmethod
    def from_absolute(size: int, step: int) -> WindowExtractor:
        def window_extractor(shape: Tuple[int, int]) -> List[BoundingBox]:
            return list(regular_grid(shape=shape, size=(size, size), step=(step, step)))

        return window_extractor

    @staticmethod
    def from_relative(rel_size: float, rel_step: float) -> WindowExtractor:
        def window_extractor(shape: Tuple[int, int]) -> List[BoundingBox]:
            size = int(rel_size * max(*shape))
            step = int(rel_step * size)
            return list(regular_grid(shape=shape, size=(size, size), step=(step, step)))

        return window_extractor

    @staticmethod
    def concat(window_extractors: List[WindowExtractor]) -> WindowExtractor:
        def window_extractor(shape: Tuple[int, int]) -> List[BoundingBox]:
            return list(
                chain.from_iterable([window_extractor(shape) for window_extractor in window_extractors])
            )

        return window_extractor
