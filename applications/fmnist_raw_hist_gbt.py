import numpy
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from mlg_lib.data.img_label_databunch import ImgLabelDatabunch
from mlg_lib.ml.ensemble_update import ensemble_update, hist_gbt_update
from mlg_lib.ml.sk_train import sk_train
from mlg_lib.transforms.flatten import flatten
# explicitly require this experimental feature
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# now you can import normally from ensemble
from sklearn.ensemble import HistGradientBoostingClassifier



if __name__ == '__main__':
    DATA_PATH = "/home/matthieu/Workspace/data/fashion_mnist.npz"

    LABEL_NAMES = ["top", "trouser", "pullover", "dress", "coat",
	"sandal", "shirt", "sneaker", "bag", "ankle boot"]

    data = numpy.load(DATA_PATH)

    bunch = ImgLabelDatabunch.from_arrays(
        ((data["xtrain"].astype(numpy.float32), data["ytrain"]), (data["xtest"].astype(numpy.float32), data["ytest"])),
        img_transform=flatten,
        names=LABEL_NAMES,
        batch_size=int(data["xtrain"].shape[0] / 5)
    )

    # fit

    training_output = sk_train(
        bunch,
        HistGradientBoostingClassifier(max_depth=2, max_iter=1, max_bins=16),
        update_fn=hist_gbt_update(10),
        metrics=[metrics.accuracy_score])(2)

    print(training_output)
