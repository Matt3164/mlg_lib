from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Callable

import numpy

from iputils.data.tensor_list import _TensorList

FileLike = Union[str, Path]
Reader = Callable[[FileLike], numpy.ndarray]

@dataclass
class FileList(_TensorList):
    files: List[FileLike]
    reader: Reader

    def __getitem__(self, item) -> numpy.ndarray:
        return self.reader(self.files[item])

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def from_path(folder: Path, glob: str, reader: Reader) -> 'FileList':
        return FileList(files=list(folder.glob(glob)), reader=reader)

    @staticmethod
    def with_suffix(file_list: 'FileList', suffix: str, reader: Reader):
        return FileList(files=[p.with_suffix(suffix) for p in file_list.files], reader=reader)
