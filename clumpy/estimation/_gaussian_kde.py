from scipy.stats import multivariate_normal
from scipy.stats import norm
from tqdm import tqdm
import itertools
import numpy as np
import multiprocessing as mp

from ._estimator import BaseEstimator

class GaussianKDE(BaseEstimator):
    """
    Gaussian Kernel Density Estimator.

    Parameters
    ----------
    s : array-like of shape (n_features)
        The diagonal of the gaussian kernel covariance matrix.
    """
    def __init__(self, s):
        self.s = s

    def fit(self, X):
        """
        fit the estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.
        """
        self.mu_ = X
        self.sigma_ = np.diag(self.s)

    def _pdf(self, X, verbose=0):
        """
        Get the probability density function.
        """
        p = np.zeros(X.shape[0])

        if verbose > 0:
            list_mu = tqdm(self.mu_)
        else:
            list_mu = self.mu_

        for mu in list_mu:
            p += multivariate_normal.pdf(X, mean=mu, cov=self.sigma_)
        p /= len(self.mu_)
        return (p)

    def _mpdf(self, X, columns, verbose=0):
        """
        Get the marginal probability density function according to
        some columns.
        """
        p = np.zeros(X.shape[0])
        if len(columns) == 1:
            cov = self.sigma_[columns[0], columns[0]]
        else:
            cov = np.diag(np.diag(self.sigma_)[columns])

        if verbose > 0:
            list_mu = tqdm(self.mu_)
        else:
            list_mu = self.mu_

        for mu in list_mu:
            mean = mu[columns]
            p += multivariate_normal.pdf(X, mean=mean, cov=cov)

        p /= len(self.mu_)

        return(p)

    def _cmpdf(self, X, column, verbose=0):
        """
        works only for one column
        """
        cdf = np.zeros(X.shape[0])

        if verbose > 1:
            list_mu = tqdm(self.mu_)
        else:
            list_mu = self.mu_

        for mu in list_mu:
            cdf += norm.cdf(X[:,0],
                            loc=mu[column],
                            scale=np.sqrt(self.sigma_[column, column]))

        return (cdf / len(self.mu_))

    def _ccpdf(self, X1, X2, column_X1, columns_X2, verbose=0):
        """
        works only for X1.shape[1] == 1 !
        a renseigne toutes les dimensions
        """

        if len(columns_X2) == 1:
            cov_X2 = self.sigma_[columns_X2[0], columns_X2[0]]
        else:
            cov_X2 = np.diag(np.diag(self.sigma_)[columns_X2])

        num = np.zeros(X1.shape[0])
        den = np.zeros(X1.shape[0])

        if verbose > 1:
            list_mu = tqdm(self.mu_)
        else:
            list_mu = self.mu_

        for mu in list_mu:
            norm_x2 = multivariate_normal.pdf(X2,
                                          mean=mu[columns_X2],
                                          cov=cov_X2)

            norm_cdf_x1 = norm.cdf(X1[:,0], mu[column_X1], np.sqrt(self.s[column_X1]))
            num += norm_x2 * norm_cdf_x1

            den += norm_x2

        return(num / den)
