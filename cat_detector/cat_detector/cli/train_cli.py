from pathlib import Path

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from pypurr.common.config import RUN_ID
from pypurr.common.helpers.model import SklearnHelpers
from pypurr.deprecated.iotools.dataset import SkDataset
from pypurr.train.model import make_classifier


def _train(
        dataset_path: Path,
        model_path: Path,
    ):
    dataset = SkDataset.from_path(dataset_path)
    X, Y = dataset.x, dataset.y

    encoder = LabelEncoder()
    Y = encoder.fit_transform(Y)

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2)

    clf = make_classifier()

    clf.fit(xtrain, ytrain.ravel())

    for tag, x, y in [
        ("train", xtrain, ytrain), ("test", xtest, ytest)]:

        print("--- Fold {} ----".format(tag))

        for val_metric in [confusion_matrix, accuracy_score]:

            print("{0}  --> {1}".format(val_metric.__name__, val_metric(y, clf.predict(x))))

    SklearnHelpers.to_path(model_path, clf)
    SklearnHelpers.to_path(model_path.parent / "encoder.pkl", encoder)


if __name__ == '__main__':

    _train(RUN_ID)
