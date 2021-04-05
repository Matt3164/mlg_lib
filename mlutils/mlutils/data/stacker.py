from mlutils.data.databunch import Databunch
from mlutils.data.numpy_dataset import NumpyDatabunch


def stack_databunch(databunch: Databunch) -> NumpyDatabunch:
    return NumpyDatabunch(
        train=databunch.train_dl.stack_one_epoch(),
        test=databunch.test_dl.stack_one_epoch(),
        valid=databunch.valid_dl.stack_one_epoch() if databunch.valid_dl else None
    )
