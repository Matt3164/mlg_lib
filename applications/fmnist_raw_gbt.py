import numpy
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier

from mlg_lib.data.img_label_databunch import ImgLabelDatabunch
from mlg_lib.ml.ensemble_update import ensemble_update
from mlg_lib.ml.sk_train import sk_train
from mlg_lib.transforms.flatten import flatten

if __name__ == '__main__':
    DATA_PATH = "/home/matthieu/Workspace/data/fashion_mnist.npz"

    LABEL_NAMES = ["top", "trouser", "pullover", "dress", "coat",
	"sandal", "shirt", "sneaker", "bag", "ankle boot"]

    data = numpy.load(DATA_PATH)

    bunch = ImgLabelDatabunch.from_arrays(
        ((data["xtrain"], data["ytrain"]), (data["xtest"], data["ytest"])),
        img_transform=flatten,
        names=LABEL_NAMES,
        batch_size=int(data["xtrain"].shape[0] / 5)
    )

    # fit

    training_output = sk_train(
        bunch,
        GradientBoostingClassifier(n_estimators=1, max_depth=3, max_features="log2", subsample=0.1),
        update_fn=ensemble_update(10),
        metrics=[metrics.accuracy_score])(10)

    print(training_output)