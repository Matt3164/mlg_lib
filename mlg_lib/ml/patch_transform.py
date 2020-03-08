import numpy as np
from skimage.util import view_as_windows
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.image import PatchExtractor


class PatchTransform(TransformerMixin, BaseEstimator):
    """"""

    def __init__(self, transformer: BaseEstimator,
                 patch_size: int,
                 max_patches: int,
                 stride: int):
        """Constructor for ProbFromClf"""
        self.transformer = transformer
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.stride = stride


    def fit(self, X, y=None):
        extractor = PatchExtractor(patch_size=(self.patch_size, self.patch_size), max_patches=self.max_patches)

        patch_X = extractor.transform(X)

        n_patches, _, _, = patch_X.shape

        self.transformer.fit(patch_X.reshape(n_patches, -1))

        return self

    def _extract_patch(self, arr):
        win_arr = view_as_windows(arr, window_shape=self.patch_size, step=self.stride)

        n_patches_i, n_patches_j, _, _ = win_arr.shape

        return self.transformer.transform(win_arr.reshape(n_patches_i * n_patches_j, -1)).flatten()

    def transform(self, X):
        return np.array([self._extract_patch(x) for x in X])

    def get_params(self, deep=True):
        return dict(transformer=self.transformer,
                    stride=self.stride,
                    patch_size=self.patch_size,
                    max_patches=self.max_patches)

    def set_params(self, **params):

        self.max_patches = params["max_patches"]
        self.stride = params["stride"]
        self.patch_size = params["patch_size"]

        params.pop("max_patches")
        params.pop("stride")
        params.pop("patch_size")

        new_kwargs = dict()

        for key in params.keys():
            if "transformer" in key:
                new_kwargs[key[len("transformer__"):]]=params[key]

        return self.transformer.set_params(**new_kwargs)