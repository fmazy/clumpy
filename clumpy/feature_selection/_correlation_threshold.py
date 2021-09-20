import numpy as np
import pandas as pd

from ._feature_selector import FeatureSelector

class CorrelationThreshold(FeatureSelector):
    """
    Feature selector that removes to closely correlated features.

    Parameters
    ----------
    threshold : float, default=1.0
        One feature among two will be removed if their correlation is greater than this threshold. The default is to keep all strictly different features.
    """
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        Learn from X.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Sample vectors from which to compute correlations.

        y : any, default=None
            Ignored. This parameter exists only for compatibility with sklearn.pipeline.Pipeline.

        Returns
        -------

        self
        """
        df = pd.DataFrame(X)

        corr = df.corr().values

        selected_features = list(np.arange(corr.shape[0]))

        corr_tril = np.abs(corr)
        corr_tril = np.tril(corr_tril) - np.diag(np.ones(corr_tril.shape[0]))

        pairs = np.where(corr_tril > self.threshold)

        features_pairs = [(pairs[0][i], pairs[1][i]) for i in range(pairs[0].size)]

        excluded_features = []

        for f0, f1 in features_pairs:
            f0_mean = np.abs(corr[:, f0]).mean()
            f1_mean = np.abs(corr[:, f1]).mean()

            if f0_mean >= f1_mean:
                feature_to_remove = f0

            else:
                feature_to_remove = f1

            excluded_features.append(feature_to_remove)
            selected_features.remove(feature_to_remove)

            # toutes les paires concernées sont retirées
            for g0, g1 in features_pairs:
                if g0 == feature_to_remove or g1 == feature_to_remove:
                    features_pairs.remove((g0, g1))

        self._cols_support = selected_features
