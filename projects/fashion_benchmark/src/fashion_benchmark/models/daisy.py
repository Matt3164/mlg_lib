from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

from mlg_lib.image.features.daisy_factory import DaisyFactory
from mlg_lib.ml.lambda_row import LambdaRow

daisy = make_pipeline(
        LambdaRow(DaisyFactory(orientations=8, histograms=8, radius=7, step=3, rings=2)),
        RandomForestClassifier(max_depth=None, max_features="log2", n_estimators=100)
    )