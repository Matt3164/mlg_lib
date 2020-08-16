import numpy
from matplotlib import pyplot
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from mlg_lib.data.image_list import _ImageList, ImageList
from mlg_lib.data.img_label_databunch import ImgLabelDatabunch
from mlg_lib.data.img_label_dataloader import ImgLabelDataloader
from mlg_lib.data.img_lbl_dataset import ImgLabelDataset
from mlg_lib.ml.ensemble_update import ensemble_update
from mlg_lib.ml.sk_train import sk_train
from mlg_lib.transforms.flatten import flatten

if __name__ == '__main__':
    DATA_PATH = "/home/matthieu/Workspace/data/fashion_mnist.npz"

    LABEL_NAMES = ["top", "trouser", "pullover", "dress", "coat",
	"sandal", "shirt", "sneaker", "bag", "ankle boot"]

    data = numpy.load(DATA_PATH)

    # ImageList example

    ImageList.from_array(data["xtrain"]).show_batch(8)

    # ImgLabelDataset

    dataset = ImgLabelDataset.from_arrays(x=data["xtrain"], y=data["ytrain"], names=LABEL_NAMES)
    dataset.show_batch(8)

    pyplot.show()

    # Raw DataLoader

    for batch_idx, batch in enumerate(ImgLabelDataloader(dataset=dataset, batch_size=10000)):
        print(f"Batch X: {batch[0].shape}, Y: {batch[1].shape}")

    # Flat pixels

    dataset = ImgLabelDataset.from_arrays(x=data["xtrain"], y=data["ytrain"], names=LABEL_NAMES, img_transform=flatten)

    for batch_idx, batch in enumerate(ImgLabelDataloader(dataset=dataset, batch_size=10000)):
        print(f"Batch X: {batch[0].shape}, Y: {batch[1].shape}")

    # ImgLabelDatabunch

    bunch = ImgLabelDatabunch.from_arrays(
        ((data["xtrain"], data["ytrain"]), (data["xtest"], data["ytest"])),
        img_transform=flatten,
        names=LABEL_NAMES,
        batch_size=int(data["xtrain"].shape[0] / 5)
    )
    print("Epoch 0")
    for (x,y) in bunch.train_dl:
        print(x.shape)

    print("Epoch 1")
    for (x, y) in bunch.train_dl:
        print(x.shape)

    # fit

    training_output = sk_train(
        bunch,
        RandomForestClassifier(n_estimators=1, max_depth=3, max_features="log2"),
        update_fn=ensemble_update(10),
        metrics=[metrics.accuracy_score])(2)

    print(training_output)
