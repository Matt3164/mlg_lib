from pathlib import Path

import numpy
from imgaug import BoundingBox
from matplotlib import pyplot

from cat_detector.config import SCALES, MIN_CONFIDENCE
from cat_detector.dataset import DatasetHelpers
from cat_detector.sklearn_helpers import SklearnHelpers
from cat_detector.preprocessing import features_fn
from cat_detector.sliding_window import SlidingWindow

if __name__ == '__main__':

    detection_dataset = DatasetHelpers.from_path(Path("/DATADRIVE1/mlegoff/workspace/data/cats/raw"))

    model = SklearnHelpers.clf_from_path("/DATADRIVE1/mlegoff/workspace/data/cats/models/model_0.pkl")

    idx = 20

    det_item = detection_dataset[idx]

    _img = det_item.x
    _y = det_item.y
    true_bbox = BoundingBox(_y[0], _y[2], _y[1], _y[3])

    window_extractor = SlidingWindow.concat(
        [SlidingWindow.from_relative(s, 0.25) for s in SCALES]
    )

    bboxes = window_extractor((_img.shape[0], _img.shape[1]))
    features = numpy.asarray([features_fn(_bb.extract_from_image(_img)) for _bb in bboxes])
    print(features.shape)
    detections = model.predict_proba(features)[:, 1] > MIN_CONFIDENCE
    detection_bboxes = [bboxes[_] for _ in range(len(bboxes)) if detections[_]]

    _display = true_bbox.draw_on_image(_img, copy=True, color=(0, 255, 0))
    for _dbbox in detection_bboxes:
        _display = _dbbox.draw_on_image(_display, color=(255, 0, 0))

    pyplot.imshow(_display)
    pyplot.show()
