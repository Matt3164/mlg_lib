from tqdm import tqdm

from data import get_fmnist, label_names
from features import Features
from iputils.data.image_list import ImageList
from iputils.data.img_lbl_dataset import ImgLabelDataset
from iputils.data.label_list import LabelList
from mlutils.data.dataset_helpers import DatasetHelpers
from mlutils.data.numpy_dataset import NumpyDatabunch
from mlutils.data.stacker import stack_databunch
from mlutils.metrics.sk_metric import make_accuracy_metric
from mlutils.training.learner import learner
from models import Models
from train import _train


def benchmark():
    logs = list()

    X, Y = get_fmnist()

    global_dataset = ImgLabelDataset(
        img_list=ImageList.from_array(X),
        lbl_list=LabelList.from_array(Y, label_names),
    )

    splitter = DatasetHelpers.splitter(train_size=2000, test_size=2000)
    global_bunch = splitter(global_dataset)

    for f in Features:

        cache_path = f"/tmp/dataset_{f.name}" + "_{}.npz"

        if NumpyDatabunch.exists(cache_path):
            bunch = NumpyDatabunch.from_file(cache_path)
        else:
            global_bunch.train_dl.dataset.dataset.img_transform = f.get_fn()
            global_bunch.test_dl.dataset.dataset.img_transform = f.get_fn()

            bunch = stack_databunch(global_bunch)
            bunch.to_file(cache_path)

        for m in tqdm(Models):

            fit_fn = learner(
                model=m.get_model(), bunch=bunch, metrics=dict(accuracy=make_accuracy_metric())
            )
            output = fit_fn()
            _logs = _train(f, m)

            logs.append(
                dict(features=f.name, model=m.name, output=output.logs)
            )

    print(logs)


if __name__ == '__main__':
    benchmark()
