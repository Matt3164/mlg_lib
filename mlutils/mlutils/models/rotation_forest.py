from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier


def make_rotation_forest(n_estimators: int = 100, n_components: int=8):
    return BaggingClassifier(
        base_estimator=make_pipeline(
            PCA(n_components=8),
            DecisionTreeClassifier()
        ),
        n_estimators=n_estimators,
        max_samples=1./n_estimators

    )
