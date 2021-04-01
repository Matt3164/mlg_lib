from itertools import product

from tqdm import tqdm

import models
from features import Features
from models import Models
from train import _train


def benchmark():
    logs = list()

    # product(Features, models)
    items = list(product(Features, [Models.rf, Models.ext]))

    for f, m in tqdm(items):
        _logs = _train(f, m)

        logs.append(
            dict(features=f.name, model=m.name, output=_logs)
        )

    print(logs)


if __name__ == '__main__':
    benchmark()
