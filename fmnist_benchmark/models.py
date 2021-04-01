from enum import Enum, auto
# explicitly require this experimental feature
from sklearn.decomposition import PCA
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# now you can import normally from ensemble
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
    HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


class Models(Enum):
    rf = auto()
    gbt = auto()
    hist_gbt = auto()
    ext = auto()
    pca_nn = auto()

    def get_model(self):
        if self == Models.rf:
            return RandomForestClassifier(bootstrap=True)
        if self == Models.gbt:
            return GradientBoostingClassifier(subsample=0.1)
        if self == Models.ext:
            return ExtraTreesClassifier(max_features=1, max_samples=0.1)
        if self == Models.hist_gbt:
            return HistGradientBoostingClassifier(max_depth=3, max_bins=32)
        if self == Models.pca_nn:
            pca = PCA(n_components=8)
            nn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
            return Pipeline(steps=[('pca', pca), ('nn', nn)])
        else:
            raise NotImplementedError