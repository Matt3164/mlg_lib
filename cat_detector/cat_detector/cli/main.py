from pathlib import Path

from pypurr.cli.build_dataset import _build_dataset
from pypurr.cli.prepare import _prepare
from pypurr.cli.preview_dataset import _preview_object_detection_dataset, _preview_classif_dataset
from pypurr.cli.scan_iou import _scan_iou
from pypurr.cli.train_cli import _train
from pypurr.common.config import IOU_THRESHOLD

if __name__ == '__main__':

    WORKSPACE = Path("/DATADRIVE1/mlegoff/workspace/data/cats/raw")
    DATA_PATH = Path("/DATADRIVE1/mlegoff/workspace/data/cats/raw")
    EXP_NAME = "exp_2407"
    IMAGE_DF = "images.csv"

    if not (WORKSPACE / IMAGE_DF).exists():
        _prepare(
            DATA_PATH,
            WORKSPACE / IMAGE_DF
        )

    # _preview_object_detection_dataset(WORKSPACE / IMAGE_DF)

    positives_folder = WORKSPACE / "new_positives"

    if not (positives_folder.exists()):
        _scan_iou(
            WORKSPACE / IMAGE_DF,
            positives_folder,
            iou_callback=lambda iou: iou >= IOU_THRESHOLD
        )

    negatives_folder = WORKSPACE / "new_negatives"
    if not (negatives_folder.exists()):
        _scan_iou(
            WORKSPACE / IMAGE_DF,
            negatives_folder,
            iou_callback=lambda iou: iou < IOU_THRESHOLD
        )

    # _preview_classif_dataset(WORKSPACE)

    dataset_file = WORKSPACE / "dataset_0.npz"

    if not dataset_file.exists():
        _build_dataset(
            WORKSPACE,
            dataset_file
        )

    model_file = WORKSPACE / "model_0.pkl"

    if not model_file.exists():
        _train(
            dataset_file,
            model_file
        )

