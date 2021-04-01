from typing import Callable

import numpy

FeaturesComputer = Callable[[numpy.ndarray], numpy.ndarray]