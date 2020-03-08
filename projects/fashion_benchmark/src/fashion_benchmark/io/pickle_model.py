import pickle

from kedro.io import AbstractDataSet
from pathlib import Path
from typing import Any, Dict


class PickleModel(AbstractDataSet):
    """"""

    def __init__(self,
                 filepath: str):
        """Constructor for NumpyDataset"""
        self._filepath = filepath

    def _load(self) -> Any:
        with open(self._filepath, "rb") as f:
            model = pickle.load(f)
        return model

    def _save(self, data: Any) -> None:
        with open(self._filepath, "wb") as f:
            pickle.dump(data, f)

    def _exists(self) -> bool:
        return Path(self._filepath).exists()

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
        )




