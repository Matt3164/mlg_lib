from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def make_classifier() -> Pipeline:
    return DecisionTreeClassifier(
        max_depth=None,
        max_features=1
    )
