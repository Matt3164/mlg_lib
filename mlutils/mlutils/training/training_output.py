from dataclasses import dataclass
from typing import Dict, Any

from sklearn.pipeline import Pipeline


@dataclass
class TrainingOutput(object):
    model: Pipeline
    logs: Dict[str, Any]
