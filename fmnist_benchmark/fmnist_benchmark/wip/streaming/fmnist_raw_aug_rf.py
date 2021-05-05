import imgaug
import numpy
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from toolz import compose

from mlg_lib_deprecated.mlg_lib import ImgLabelDatabunch
from mlg_lib_deprecated.mlg_lib import ensemble_update
from mlg_lib_deprecated.mlg_lib import sk_train
from iputils.features.flatten import flatten

if __name__ == '__main__':
    DATA_PATH = "/home/matthieu/Workspace/data/fashion_mnist.npz"

    LABEL_NAMES = ["top", "trouser", "pullover", "dress", "coat",
	"sandal", "shirt", "sneaker", "bag", "ankle boot"]

    data = numpy.load(DATA_PATH)

    # Interesting to see that crop and pad does not seem to help compare to flip

    seq = imgaug.augmenters.Sequential(
        [
            imgaug.augmenters.Fliplr(0.5),
            #imgaug.augmenters.CropAndPad(px=(-2, 2)),
            #imgaug.augmenters.PadToFixedSize(width=28, height=28),
            #imgaug.augmenters.CropToFixedSize(width=28, height=28),
        ]
    )

    bunch = ImgLabelDatabunch.from_arrays(
        ((data["xtrain"], data["ytrain"]), (data["xtest"], data["ytest"])),
        img_transform=compose(flatten, seq.augment_image),
        names=LABEL_NAMES,
        batch_size=int(data["xtrain"].shape[0] / 5)
    )

    # fit

    training_output = sk_train(
        bunch,
        RandomForestClassifier(n_estimators=1, max_depth=3, max_features="log2"),
        update_fn=ensemble_update(5),
        metrics=[metrics.accuracy_score])(4)

    print(training_output)
