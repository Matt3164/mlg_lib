from copy import deepcopy
from typing import Callable, Any, List

import numpy
import numpy as np
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from mlg_lib.data.img_label_databunch import ImgLabelDatabunch
from mlg_lib.ml.training_output import TrainingOutput


def sk_train(
    bunch: ImgLabelDatabunch,
    model: Pipeline,
    update_fn: Callable[[Pipeline, numpy.ndarray, numpy.ndarray], Pipeline],
    metrics: List[Callable[[np.ndarray, np.ndarray], Any]]=list()
    )->Callable[[int], TrainingOutput]:

    def fit_fn(n_epochs: int)->TrainingOutput:
        sk_model = deepcopy(model)

        for epoch in tqdm(range(n_epochs), desc="Epochs"):
            for (x,y) in bunch.train_dl:
                sk_model = update_fn(sk_model, x, y)

        computed_metrics = dict()

        for metric_fn in metrics:
            for tag, dl in [("train", bunch.train_dl ), ("test", bunch.valid_dl)]:
                batch_metrics = [metric_fn(y, sk_model.predict(x)) for x,y in dl]

                computed_metrics["_".join([tag, metric_fn.__name__])] = numpy.mean(batch_metrics)

        return TrainingOutput(
            models=sk_model,
            metrics=computed_metrics
        )
    return fit_fn