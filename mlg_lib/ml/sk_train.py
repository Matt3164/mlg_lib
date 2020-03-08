from typing import Dict, Callable, Any

import numpy as np
from sklearn.pipeline import Pipeline

from mlg_lib.ml.training_output import TrainingOutput


def sk_train(
    xtrain: np.ndarray,
    xtest: np.ndarray,
    ytrain: np.ndarray,
    ytest: np.ndarray,
    model: Pipeline,
    metrics: Dict[str, Callable[[np.ndarray, np.ndarray], Any]]
    )->TrainingOutput:

    model.fit(xtrain, ytrain)

    computed_metrics = dict()

    for tag, (x,y) in [("train", (xtrain, ytrain) ), ("test", (xtest, ytest))]:
        predictions = model.predict(x)

        for metric_tag, metric_fn in metrics:
            computed_metrics["_".join([tag, metric_tag])] = metric_fn(y, predictions)

    return TrainingOutput(
        model=model,
        metrics=computed_metrics
    )