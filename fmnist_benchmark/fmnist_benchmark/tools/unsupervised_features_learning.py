import numpy
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from mlutils.sklearn_utils import model_to_path
from fmnist_benchmark.data import get_fmnist


def _relu(x):
    x[x < 0] = 0
    return x

def unsup_feat_learning():
    X, Y = get_fmnist()

    patches = list()
    # for idx in range(X.shape[0]):
    patch_size = 7
    n_max_images = 100

    for idx in range(n_max_images):
        item = X[idx, ::]

        patches.append(extract_patches_2d(item, patch_size=(patch_size, patch_size), max_patches=10))

    patches = numpy.vstack(patches)

    print(patches.shape)

    x = patches.reshape(-1, patch_size * patch_size)

    pipeline = make_pipeline(
            MiniBatchKMeans(n_init=1, n_clusters=256, max_iter=1),
            StandardScaler(),
            FunctionTransformer(_relu, validate=False)
    )

    pipeline.fit(x)

    print(pipeline.steps[0][1].cluster_centers_.shape)

    numpy.save("/tmp/fmnist_bench/clusters.npy", pipeline.steps[0][1].cluster_centers_)

    model_to_path(pipeline, "/tmp/fmnist_bench/km.pickle")


if __name__ == '__main__':
    unsup_feat_learning()
