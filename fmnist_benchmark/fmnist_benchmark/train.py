from typing import Dict

import click

from fmnist_benchmark.data import get_fmnist, label_names
from fmnist_benchmark.features import Features
from iputils.data.image_list import ImageList
from iputils.data.img_lbl_dataset import ImgLabelDataset
from iputils.data.label_list import LabelList
from mlutils.data.dataset_helpers import DatasetHelpers
from mlutils.data.stacker import stack_databunch
from mlutils.metrics.sk_metric import make_accuracy_metric
from mlutils.training.learner import learner
from fmnist_benchmark.models import Models


@click.command()
@click.option("--feature", type=click.Choice([f.name for f in Features]), default='raw')
@click.option("--model", type=click.Choice([f.name for f in Models]), default='rf')
@click.option("--debug", type=click.BOOL, default=False, is_flag=True)
def train_cli(feature: str, model: str, debug: bool):
    model = Models[model]
    feature = Features[feature]
    _train(feature, model, debug)


def _train(feature: Features, model: Models, debug: bool = False) -> Dict:
    model = model.get_model()
    X, Y = get_fmnist()

    global_dataset = ImgLabelDataset(
        img_list=ImageList.from_array(X),
        lbl_list=LabelList.from_array(Y, label_names),
        img_transform=feature.get_fn()
    )

    if debug:
        splitter = DatasetHelpers.splitter(train_size=2000, test_size=2000)
    else:
        splitter = DatasetHelpers.splitter(train_size=0.8, test_size=0.2)

    bunch = stack_databunch(splitter(global_dataset))

    fit_fn = learner(
        model=model, bunch=bunch, metrics=dict(accuracy=make_accuracy_metric())
    )
    output = fit_fn()

    return output.logs


if __name__ == '__main__':
    train_cli()
