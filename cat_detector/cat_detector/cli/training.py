from pathlib import Path

import numpy

from cat_detector.sklearn_helpers import SklearnHelpers
from cat_detector.preprocessing import features_fn
from fmnist_benchmark.models import Models
from iputils.data.image_list import ImageList
from iputils.data.img_lbl_dataset import ImgLabelDataset
from iputils.data.label_list import LabelList
from mlutils.data.dataset_helpers import DatasetHelpers
from mlutils.data.stacker import stack_databunch
from mlutils.metrics.sk_metric import make_accuracy_metric
from mlutils.training.learner import learner


if __name__ == '__main__':

    model_path = Path("/DATADRIVE1/mlegoff/workspace/data/cats/models")
    model_path.mkdir(exist_ok=True)
    img_list = ImageList.from_path(Path("/DATADRIVE1/mlegoff/workspace/data/cats/classif"), "*/*.png")

    labels = numpy.asarray([1 if "positives" in str(f) else 0 for f in img_list.files])

    dataset = ImgLabelDataset(
        img_list=img_list,
        lbl_list=LabelList.from_array(labels, names=["not a cat", "cat"]),
        img_transform=features_fn
    )

    splitter = DatasetHelpers.splitter(train_size=0.8, test_size=0.2)

    bunch = stack_databunch(splitter(dataset))

    fit_fn = learner(
        model=Models.ext.get_model(), bunch=bunch, metrics=dict(accuracy=make_accuracy_metric())
    )
    output = fit_fn()

    SklearnHelpers.to_path(model_path / "model_0.pkl", output.model)
