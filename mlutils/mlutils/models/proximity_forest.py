from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline


def make_proximity_forest(n_exemplars: int = 256):
    return make_pipeline(
        MiniBatchKMeans(n_init=1, n_clusters=n_exemplars, max_iter=1),
        RandomForestClassifier(bootstrap=True)
    )
