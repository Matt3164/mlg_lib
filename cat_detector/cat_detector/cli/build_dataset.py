from pathlib import Path

from pypurr.common.config import RUN_ID
from pypurr.common.preprocessing.window import resize_to_gray_flatten
from pypurr.deprecated.iotools.dataset import ImageClassificationDatasetHelpers
from pypurr.train.helpers.dataset.classif import ImageClassificationDataset


def _build_dataset(
        img_folder: Path,
        out_npz: Path
    ) -> None:

    dataset = ImageClassificationDataset.from_folders(img_folder, features_fn=resize_to_gray_flatten)

    ImageClassificationDatasetHelpers.to_sk_dataset(dataset).to_path(out_npz)



if __name__ == '__main__':

    _build_dataset(RUN_ID)
