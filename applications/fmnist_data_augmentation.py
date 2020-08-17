from itertools import islice

import imgaug
import numpy
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from toolz import identity

from mlg_lib.data.img_label_databunch import ImgLabelDatabunch

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

    seq = imgaug.augmenters.Sequential(
        [
            imgaug.augmenters.Fliplr(0.5),
            imgaug.augmenters.CropAndPad(px=(-2,2)),
            imgaug.augmenters.PadToFixedSize(width=28,height=28),
            imgaug.augmenters.CropToFixedSize(width=28, height=28),
            # imgaug.augmenters.Multiply(mul=(0.80, 1.)),
            # imgaug.augmenters.Add(value=(-10, 10)),
            # imgaug.augmenters.AdditiveGaussianNoise(scale=(10, 20)),
            # imgaug.augmenters.CoarseDropout(p=0.2, size_percent=0.05)
        ]
    )



    bunch = ImgLabelDatabunch.from_arrays(
        ((xtrain, ytrain), (xtest, ytest)),
        img_transform=lambda _: seq.augment_image(_),
        names=LABEL_NAMES,
        batch_size=int(xtrain.shape[0] / 4)
    )

    n = 4

    for (x,y) in islice(bunch.train_dl, None, 1):

        grid_sz = n

        for _ in range(n**2):
            pyplot.subplot(grid_sz, grid_sz, _ + 1)
            pyplot.imshow(x[_,::], cmap="gray")


    pyplot.show()
