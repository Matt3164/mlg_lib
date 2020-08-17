from itertools import islice

import numpy
from matplotlib import pyplot
from sklearn.decomposition import PCA

from mlg_lib.data.img_label_databunch import ImgLabelDatabunch
from mlg_lib.transforms.flatten import flatten

if __name__ == '__main__':
    DATA_PATH = "/home/matthieu/Workspace/data/fashion_mnist.npz"

    LABEL_NAMES = ["top", "trouser", "pullover", "dress", "coat",
	"sandal", "shirt", "sneaker", "bag", "ankle boot"]

    data = numpy.load(DATA_PATH)

    # ImgLabelDatabunch

    bunch = ImgLabelDatabunch.from_arrays(
        ((data["xtrain"], data["ytrain"]), (data["xtest"], data["ytest"])),
        img_transform=flatten,
        names=LABEL_NAMES,
        batch_size=int(data["xtrain"].shape[0]/1)
    )

    n = 4

    for (x,y) in islice(bunch.train_dl, None, 1):

        pca = PCA(n_components=n**2, whiten=True)
        pca.fit(x)

        pyplot.figure()
        for i in range(n**2):
            pyplot.subplot(n,n,i+1)
            pyplot.imshow(pca.components_[i,:].reshape(28,28))

    pyplot.show()

