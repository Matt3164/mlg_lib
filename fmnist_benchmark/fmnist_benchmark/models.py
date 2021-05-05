from enum import Enum, auto

# explicitly require this experimental feature
from sklearn.decomposition import PCA
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# now you can import normally from ensemble
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
    HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from mlutils.models.multiboosting import make_multiboosting
from mlutils.models.proximity_forest import make_proximity_forest
from mlutils.models.rotation_forest import make_rotation_forest


class Models(Enum):
    rf = auto()
    gbt = auto()
    hist_gbt = auto()
    ext = auto()
    pca_nn = auto()
    multiboosting = auto()
    proximity_forest = auto()
    rotation_forest = auto()
    logistic = auto()
    ridge = auto()

    def get_model(self):
        if self == Models.rf:
            return RandomForestClassifier(bootstrap=True)
        elif self == Models.gbt:
            return GradientBoostingClassifier(subsample=0.1)
        elif self == Models.ext:
            return ExtraTreesClassifier(max_features=1, max_samples=0.1)
        elif self == Models.hist_gbt:
            return HistGradientBoostingClassifier(max_depth=3, max_bins=32)
        elif self == Models.pca_nn:
            pca = PCA(n_components=8)
            nn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
            return Pipeline(steps=[('pca', pca), ('nn', nn)])
        elif self == Models.multiboosting:
            return make_multiboosting()
        elif self == Models.proximity_forest:
            return make_proximity_forest(n_exemplars=1024)
        elif self == Models.rotation_forest:
            return make_rotation_forest()
        elif self == Models.logistic:
            return SGDClassifier(loss="log", max_iter=100)
        elif self == Models.ridge:
            return RidgeClassifier(normalize=True, max_iter=100)
        else:
            raise NotImplementedError
