from tqdm import tqdm
import numpy as np
from scipy.special import gamma

from ._estimator import BaseEstimator

class UniformSphereKDE(BaseEstimator):
    """
    Uniform Sphere Kernel Density Estimator.

    Parameters
    ----------
    radius : float
        The hypersphere radius
    """
    def __init__(self, radius):
        self.radius = radius



    def fit(self, X):
        """
        fit the estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.
        """
        self.mu = X
        # hypersphere volume
        s = X.shape[1]
        self._V = np.pi ** (s / 2) * self.radius ** s / gamma(s / 2 + 1)

    def _pdf(self, X, verbose=0):
        p = np.zeros(X.shape[0])

        if verbose > 0:
            list_mu = tqdm(self.mu)
        else:
            list_mu = self.mu

        for mu in list_mu:
            p += (np.linalg.norm(X-mu, axis=1) <= self.radius).astype(float)

        p /= self.mu.shape[0] * self._V
        return (p)
