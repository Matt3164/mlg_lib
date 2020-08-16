import numpy
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from mlg_lib.data.img_label_databunch import ImgLabelDatabunch
from mlg_lib.ml.ensemble_update import ensemble_update
from mlg_lib.ml.sk_train import sk_train
from mlg_lib.transforms.daisy import wrapped_daisy

if __name__ == '__main__':
    DATA_PATH = "/home/matthieu/Workspace/data/fashion_mnist.npz"

    LABEL_NAMES = ["top", "trouser", "pullover", "dress", "coat",
	"sandal", "shirt", "sneaker", "bag", "ankle boot"]

    data = numpy.load(DATA_PATH)

    bunch = ImgLabelDatabunch.from_arrays(
        ((data["xtrain"], data["ytrain"]), (data["xtest"], data["ytest"])),
        img_transform=wrapped_daisy(orientations=4),
        names=LABEL_NAMES,
        batch_size=int(data["xtrain"].shape[0] / 1)
    )

    # fit

    training_output = sk_train(
        bunch,
        RandomForestClassifier(n_estimators=1, max_depth=3, max_features="log2"),
        update_fn=ensemble_update(100),
        metrics=[metrics.accuracy_score])(1)

    print(training_output)
