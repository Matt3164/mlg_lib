from pathlib import Path

from typing import Any, Dict

from kedro.io import AbstractDataSet
import numpy as np



class NumpyDataset(AbstractDataSet):
    """"""

    def __init__(self,
                 filepath: str):
        """Constructor for NumpyDataset"""
        self._filepath = filepath

    def _load(self) -> np.ndarray:
        return np.load(self._filepath, allow_pickle=True)

    def _save(self, data: np.ndarray) -> None:
        np.save(self._filepath, data)

    def _exists(self) -> bool:
        return Path(self._filepath).exists()

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
        )




