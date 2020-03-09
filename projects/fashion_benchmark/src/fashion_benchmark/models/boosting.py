
from sklearn.pipeline import make_pipeline

from mlg_lib.image.features.hog_factory import HogFactory
from mlg_lib.ml.lambda_row import LambdaRow

# explicitly require this experimental feature
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# now you can import normally from ensemble
from sklearn.ensemble import HistGradientBoostingClassifier

#%%
hist_boosting = make_pipeline(
    LambdaRow(HogFactory(orientations=8, pixels_per_cell=(7, 7),cells_per_block=(1, 1), multichannel=False)),
    HistGradientBoostingClassifier(max_depth=2, max_iter=50)
)