from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

from mlg_lib.image.features.hog_factory import HogFactory
from mlg_lib.ml.lambda_row import LambdaRow

# orientations=8,
#                pixels_per_cell=(8, 8),
#                cells_per_block=(2, 2),
#                block_norm="L2",
#                transform_sqrt=True,
#                feature_vector=True

hog = make_pipeline(
        LambdaRow(HogFactory(orientations=8, pixels_per_cell=(8, 8),cells_per_block=(2, 2), multichannel=False)),
        RandomForestClassifier(max_depth=None, max_features="log2", n_estimators=100)
    )