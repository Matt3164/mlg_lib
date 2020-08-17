import numpy
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from mlg_lib.data.img_label_databunch import ImgLabelDatabunch
from mlg_lib.ml.ensemble_update import ensemble_update
from mlg_lib.ml.sk_train import sk_train
from mlg_lib.transforms.daisy import wrapped_daisy

if __name__ == '__main__':
    DATA_PATH = "/home/matthieu/Workspace/data/fashion_mnist.npz"

    LABEL_NAMES = ["top", "trouser", "pullover", "dress", "coat",
	"sandal", "shirt", "sneaker", "bag", "ankle boot"]

    data = numpy.load(DATA_PATH)

    xtrain = data["xtrain"]
    ytrain = data["ytrain"]
    xtest = data["xtest"]
    ytest = data["ytest"]

    xtrain, _, ytrain, _ = train_test_split(xtrain, ytrain, train_size=0.1)
    xtest, _, ytest, _ = train_test_split(xtest, ytest, train_size=0.1)

    print(f"Train {xtrain.shape}")
    print(f"Test {xtest.shape}")

    bunch = ImgLabelDatabunch.from_arrays(
        ((xtrain, ytrain), (xtest, ytest)),
        img_transform=wrapped_daisy(),
        names=LABEL_NAMES,
        batch_size=int(xtrain.shape[0] / 4)
    )

    # fit

    training_output = sk_train(
        bunch,
        RandomForestClassifier(n_estimators=1, max_depth=None, max_features="log2"),
        update_fn=ensemble_update(25),
        metrics=[metrics.accuracy_score])(1)

    print(training_output)
