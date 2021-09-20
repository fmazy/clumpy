import numpy as np

from ._feature_selector import FeatureSelector

class VarianceThreshold(FeatureSelector):
    """
    Feature selector that removes all low-variance features.

    Parameters
    ----------

    Threshold : float, default=0.0
        Features with a training-set variance lower than this threshold will be removed. The default is to keep all features with non-zero variance, i.e. remove the features that have the same value in all samples.
    """
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        Learn from X.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Sample vectors from which to compute variances.

        y : any, default=None
            Ignored. This parameter exists only for compatibility with sklearn.pipeline.Pipeline.

        Returns
        -------

        self
        """
        self._cols_support = np.where(X.var(axis=0) >= self.threshold)[0]

        return(self)
