from pathlib import Path

from matplotlib.pyplot import imshow, show, subplot, title

from pypurr.train.helpers.dataset.classif import ImageClassificationDataset
from pypurr.train.helpers.dataset.objdet import ObjectDetectionDataset


def _preview_object_detection_dataset(
        image_df_path: Path,
):
    detection_dataset = ObjectDetectionDataset.from_path(image_df_path)

    n_samples = min(len(detection_dataset), 10)
    for idx in range(n_samples):
        det_item = detection_dataset[idx]
        imshow(det_item.bbox.draw_box_on_image(det_item.img))
        show()


def _preview_classif_dataset(
        directory: Path,
):
    classif_dataset = ImageClassificationDataset.from_folders(directory, features_fn=lambda x: x)

    n_samples = min(len(classif_dataset), 16)

    for idx in range(n_samples):
        subplot(4, 4, idx + 1)
        imshow(classif_dataset.get_img(idx))
        title(classif_dataset[idx].label)
    show()
