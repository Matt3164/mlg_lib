from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

from mlg_lib.image.features.flatten import flatten
from mlg_lib.ml.lambda_row import LambdaRow

raw_pix = make_pipeline(
        LambdaRow(flatten),
        RandomForestClassifier(max_depth=None, max_features="log2", n_estimators=100)
    )