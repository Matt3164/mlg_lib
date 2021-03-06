from pathlib import Path

import numpy
from imgaug import BoundingBox
from tqdm import tqdm

from cat_detector.config import MAX_ITEMS, SCALES, IOU_THRESHOLD, MAX_NEGATIVES_BY_ITEM
from cat_detector.dataset import DatasetHelpers
from cat_detector.opencv_helpers import OpenCVHelpers
from cat_detector.sliding_window import SlidingWindow

if __name__ == '__main__':

    out_path = Path("/DATADRIVE1/mlegoff/workspace/data/cats/classif/negatives")
    out_path.mkdir(exist_ok=True)

    detection_dataset = DatasetHelpers.from_path(Path("/DATADRIVE1/mlegoff/workspace/data/cats/raw"))
    n_samples = min(len(detection_dataset), MAX_ITEMS)

    window_extractor = SlidingWindow.concat(
        [SlidingWindow.from_relative(s, 0.25) for s in SCALES]
    )

    positive_count = 0

    for idx in tqdm(range(n_samples)):
        det_item = detection_dataset[idx]
        _img = det_item.x
        bboxes = window_extractor((_img.shape[0], _img.shape[1]))
        crops = [_bb.extract_from_image(det_item.x) for _bb in bboxes]
        _y = det_item.y
        bbox = BoundingBox(_y[0], _y[2], _y[1], _y[3])
        ious = [bb.iou(bbox) for bb in bboxes]

        crop_selection = numpy.asarray([iou < IOU_THRESHOLD for iou in ious])

        selected_crops = [crops[_] for _ in range(len(crops)) if crop_selection[_]]

        idx = numpy.random.permutation(numpy.arange(len(selected_crops)))[:MAX_NEGATIVES_BY_ITEM]

        selected_crops = [selected_crops[i] for i in idx]

        for i in range(len(selected_crops)):
            OpenCVHelpers.imwrite(out_path / f"{positive_count}.png", selected_crops[i])
            positive_count += 1