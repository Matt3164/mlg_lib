from dataclasses import dataclass
from typing import Dict, Any, List

from sklearn.pipeline import Pipeline


@dataclass
class TrainingOutput(object):
    models: List[Pipeline]
    metrics : Dict[str, Any]
