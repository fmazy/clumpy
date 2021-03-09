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
    def __init__(self,
                 sigma,
                 bounded_columns=[]):

        super().__init__(bounded_columns=bounded_columns)
        self.sigma = sigma

    def fit(self, X):
        """
        fit the estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.
        """
        self.mu = X.copy()
        self.min_ = np.min(X, axis=0)

        # on ajoute des noyaux sous la barre
        idx = np.any(np.abs(self.mu[:, self.bounded_columns] - self.min_[self.bounded_columns] - self.sigma[self.bounded_columns]/ 2) <= self.sigma[self.bounded_columns] / 2, axis=1)

        print(idx.sum()/idx.size)

        mu_to_stack = self.mu[idx,:]
        for i in self.bounded_columns:
            mu_to_stack[:,i] = 2 * self.min_[i] - mu_to_stack[:,i]

        self.mu = np.vstack((self.mu, mu_to_stack))

        # self.V_ = np.zeros_like(X)
        # self.V_[:,:] = 2 * self.sigma

        # for i_feature in self.bounded_columns:
        #     idx = np.abs(self.mu[:, i_feature] - self.min_[i_feature] - self.sigma[i_feature]/ 2) <= self.sigma[i_feature] / 2
        #     self.V_[idx, i_feature] = self.mu[idx, i_feature] + self.sigma[i_feature] - self.min_[i_feature]

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
            # p += np.all(np.abs(X - mu) <= self.sigma, axis=1).astype(float)
            p_mu = np.all(np.abs(X - mu) <= self.sigma, axis=1).astype(float)
            p_mu /= self.mu.shape[0] * np.product(2*self.sigma)

            p += p_mu

        return (p)

    def _cdf(self, X, verbose=0):
        cdf = np.zeros(X.shape[0])

        if verbose > 0:
            list_mu = tqdm(self.mu)
        else:
            list_mu = self.mu

        V = np.product(2 * self.sigma)

        # V_offset = 0

        den_offset = 0

        for mu in list_mu:
            d = X - mu + self.sigma

            # threshold
            for i_feature in range(d.shape[1]):
                d[d[:,i_feature] > 2 * self.sigma[i_feature],i_feature] = 2 * self.sigma[i_feature]

            # adjustment for bounded columns

            # mu_den = V
            #
            # for i_feature in self.bounded_columns:
            #     # is it a problematic kernel center ?
            #     if np.abs(mu[i_feature] - self.min_[i_feature] - self.sigma[i_feature] / 2) <= self.sigma[i_feature] / 2:
            #         d_offset = self.min_[i_feature] - mu[i_feature] + self.sigma[i_feature]
            #
            #         d[:, i_feature] -= d_offset
            #
            #         den_offset += V * (d_offset) / (2 * self.sigma[i_feature])

            d[d < 0] = 0

            cdf += np.product(d, axis=1)

        cdf /= V * self.mu.shape[0] - den_offset

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
            p += np.all(np.abs(X - mu[columns]) <= self.sigma[columns], axis=1)

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

        V_X2 = np.product(2*self.sigma[columns_X2])

        for mu in list_mu:
            K_x2 = np.all(np.abs(X2 - mu[columns_X2]) <= self.sigma[columns_X2], axis=1) / V_X2

            cdf_K_x1 = X1[:,0] - mu[column_X1] + self.sigma[column_X1]
            cdf_K_x1[cdf_K_x1 < 0] = 0
            cdf_K_x1[cdf_K_x1 > 2 * self.sigma[column_X1]] = 2 * self.sigma[column_X1]

            num += K_x2 * cdf_K_x1

            den += K_x2
        ccpdf = np.zeros(X1.shape[0])
        id_not_null = den > 0

        ccpdf[id_not_null] = num[id_not_null] / den[id_not_null]

        return (ccpdf)


