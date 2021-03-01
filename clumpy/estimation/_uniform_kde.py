from tqdm import tqdm
import numpy as np

from ._estimator import BaseEstimator

class UniformKDE(BaseEstimator):
    """
    Gaussian Kernel Density Estimator.

    Parameters
    ----------
    s : array-like of shape (n_features)
        The distance along each direction
    """
    def __init__(self, sigma):
        self.sigma = sigma

    def fit(self, X):
        """
        fit the estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.
        """
        self.mu = X

    def _pdf(self, X, verbose=0):
        """
        Get the probability density function.
        """
        p = np.zeros(X.shape[0])

        if verbose > 0:
            list_mu = tqdm(self.mu)
        else:
            list_mu = self.mu

        for mu in list_mu:
            p += np.all(np.abs(X - mu) <= self.sigma, axis=1)
        p /= len(self.mu) * np.product(2*self.sigma)
        return (p)

    def _cdf(self, X, verbose=0):
        cdf = np.zeros(X.shape[0])

        if verbose > 0:
            list_mu = tqdm(self.mu)
        else:
            list_mu = self.mu

        V = np.product(2 * self.sigma)

        for mu in list_mu:
            idx_inside = np.all(np.abs(X - mu) <= self.sigma, axis=1)
            idx_after = np.all(X - mu > self.sigma, axis=1)

            cdf[idx_inside] += np.product(X[idx_inside,:] - mu + self.sigma, axis=1) / V

            cdf[idx_after] += 1

        cdf /= len(self.mu)

        return(cdf)

    def _mpdf(self, X, columns, verbose=0):
        """
        Get the marginal probability density function according to
        some columns.
        """
        p = np.zeros(X.shape[0])

        if verbose > 0:
            list_mu = tqdm(self.mu)
        else:
            list_mu = self.mu

        for mu in list_mu:
            p += np.all(np.abs(X[:,columns] - mu[columns]) <= self.sigma[columns], axis=1)

        p /= len(self.mu) * np.product(2*self.sigma[columns])

        return (p)

    def _cmpdf(self, X, column, verbose=0):
        X_prime = np.zeros((X.shape[0], self.mu.shape[1]))
        column_bar = np.delete(np.arange(self.mu.shape[1]), column)
        X_prime[:, column] = X[:,0]
        X_prime[:, column_bar] = np.inf

        return(self._cdf(X_prime, verbose=verbose))

    def _ccpdf(self, X1, X2, column_X1, columns_X2, verbose=0):
        """
        works only for X1.shape[1] == 1 !
        a renseigne toutes les dimensions
        """
        num = np.zeros(X1.shape[0])
        den = np.zeros(X1.shape[0])

        if verbose > 0:
            list_mu = tqdm(self.mu)
        else:
            list_mu = self.mu

        for mu in list_mu:
            K_x2 = np.all(np.abs(X2 - mu[columns_X2]) <= self.sigma[columns_X2], axis=1) / np.product(2*self.sigma[columns_X2])

            idx_inside = np.all(np.abs(X1 - mu[column_X1]) <= self.sigma[column_X1], axis=1)
            idx_after = np.all(X1 - mu[column_X1] > self.sigma[column_X1], axis=1)
            cdf_K_x1 = np.zeros(X1.shape[0])

            cdf_K_x1[idx_inside] += np.product(X1[idx_inside,:] - mu[column_X1] + self.sigma[column_X1], axis=1) / np.product(2*self.sigma[column_X1])

            cdf_K_x1[idx_after] = 1

            num += K_x2 * cdf_K_x1

            den += K_x2

        return (num / den)


