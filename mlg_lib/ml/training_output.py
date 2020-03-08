from typing import Dict, Any

import attr
from sklearn.pipeline import Pipeline


@attr.s
class TrainingOutput(object):

    model = attr.ib(type=Pipeline)
    metrics = attr.ib(type=Dict[str, Any])