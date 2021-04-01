from typing import Dict

import click

from data import get_fmnist
from features import Features, compute_features
from mlutils.data.numpy_dataset import NumpyDataset
from mlutils.data.split import make_splitter
from mlutils.metrics.sk_metric import make_accuracy_metric
from mlutils.training.learner import learner
from models import Models
from unsupervised_features_learning import _relu

@click.command()
@click.option("--feature", type=click.Choice([f.name for f in Features]), default='raw')
@click.option("--model", type=click.Choice([f.name for f in Models]), default='rf')
def train_cli(feature: str, model: str):
    model = Models[model]
    feature = Features[feature]
    _train(feature, model)


def _train(feature: Features, model: Models) -> Dict:
    model = model.get_model()
    X, Y = get_fmnist()
    global_dataset = NumpyDataset.from_arrays(X=X, Y=Y)

    subsample_splitter = make_splitter(train_size=2000)
    splitter = make_splitter(train_size=0.5, test_size=0.5)

    subsampled_dataset = subsample_splitter(global_dataset).train
    dataset = NumpyDataset.from_arrays(X=compute_features(subsampled_dataset.x, feature), Y=subsampled_dataset.y)
    print(dataset)
    bunch = splitter(dataset)

    fit_fn = learner(
        model=model, bunch=bunch, metrics=dict(accuracy=make_accuracy_metric())
    )
    output = fit_fn()

    return output.logs


if __name__ == '__main__':
    train_cli()
