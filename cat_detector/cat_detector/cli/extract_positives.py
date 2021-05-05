from pathlib import Path

from imgaug import BoundingBox
from tqdm import tqdm

from cat_detector.config import MAX_ITEMS, SCALES, IOU_THRESHOLD
from cat_detector.dataset import DatasetHelpers
from cat_detector.opencv_helpers import OpenCVHelpers
from cat_detector.sliding_window import SlidingWindow

if __name__ == '__main__':

    detection_dataset = DatasetHelpers.from_path(Path("/DATADRIVE1/mlegoff/workspace/data/cats/raw"))

    out_path = Path("/DATADRIVE1/mlegoff/workspace/data/cats/classif/positives")
    out_path.mkdir(exist_ok=True)

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

        crop_selection = [iou > IOU_THRESHOLD for iou in ious]
        selected_crops = [crops[_] for _ in range(len(crops)) if crop_selection[_]]

        for i in range(len(selected_crops)):
            OpenCVHelpers.imwrite(out_path / f"{positive_count}.png", selected_crops[i])
            positive_count += 1
