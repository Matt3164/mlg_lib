from os import makedirs
from pathlib import Path
from typing import Callable

import numpy
from skimage.util import view_as_windows

from pypurr.common.config import SCALES, IOU_THRESHOLD, MAX_POSITIVE_EXAMPLES
from pypurr.common.scanner.grid import regular_grid
from pypurr.opencv_helpers import OpenCVHelpers
from pypurr.train.helpers.dataset.objdet import ObjectDetectionDataset


def _scan_iou(
        image_df: Path,
        out_path: Path,
        iou_callback: Callable[[float], bool]

):
    makedirs(out_path, exist_ok=True)
    detection_dataset = ObjectDetectionDataset.from_path(image_df)

    n_samples = min(len(detection_dataset), 10)

    positive_count = 0

    for idx in range(n_samples):
        det_item = detection_dataset[idx]

        for s in SCALES:
            bboxes = list(regular_grid(det_item.img.shape, (s, s), (0.25 * s, 0.25 * s)))
            crops = view_as_windows(det_item.img, (s, s, 3), step=int(0.25 * s)).reshape(-1,s,s,3)
            ious = [bb.iou(det_item.bbox) for bb in bboxes]

            crop_selection = numpy.asarray([iou_callback(iou) for iou in ious])
            selected_crops = crops[crop_selection, :, :, :]

            for i in range(selected_crops.shape[0]):

                if positive_count >= MAX_POSITIVE_EXAMPLES:
                    break

                OpenCVHelpers.imwrite(out_path / f"{positive_count}.png", selected_crops[i, :, :, :])
                positive_count += 1

        if positive_count >= MAX_POSITIVE_EXAMPLES:
            break


if __name__ == '__main__':
    _scan_iou()
