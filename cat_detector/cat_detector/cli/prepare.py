from glob import glob
from itertools import chain
from os import listdir
from os.path import join
from pathlib import Path
from typing import Tuple

import numpy as np
from pandas import DataFrame


def _prepare(
        images_dir: Path,
        out_df: Path
) -> None:
    image_files = chain.from_iterable(
        map(lambda folder: glob(join(images_dir, folder, "*.jpg")), listdir(images_dir))
    )

    data = list()
    for img_path in image_files:

        bbox = _bbox_from_filepath(img_path)

        data.append(
            [img_path] + list(bbox)
        )

    df = DataFrame.from_records(data, columns=["ImagePath", "Xmin", "Xmax", "Ymin", "Ymax"])

    df.to_csv(out_df)

    return None


def get() -> None:
    """
    Should download dataset files

    https://www.kaggle.com/crawford/cat-dataset
    https://archive.org/details/CAT_DATASET
    https://archive.org/download/CAT_DATASET/CAT_DATASET_01.zip
    https://archive.org/download/CAT_DATASET/CAT_DATASET_02.zip

    Returns:

    """
    return None


def _bbox_from_filepath(image_file: str) -> Tuple[int, int, int, int]:
    with open(image_file + ".cat", "r") as f:
        content = f.read()

    pos = [int(e) for e in content.split(" ") if e != '']

    pos = np.array(pos)[1:]

    x = pos[::2]

    y = pos[1::2]

    return x.min(), x.max(), y.min(), y.max()


if __name__ == '__main__':
    _prepare()
