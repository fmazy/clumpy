from scipy.stats import multivariate_normal
from scipy.stats import norm
from tqdm import tqdm
import itertools
import numpy as np

class GaussianKDE():
    def __init__(self, s):
        self.s = s

    def fit(self, X):
        self.mu_ = X
        self.sigma_ = np.diag(self.s)

    def predict(self, X):
        return(self._pdf(X))

    def score(self, X):
        return(self.ks(X))

    def marginal_ks(self, X):
        n_samples = X.shape[0]

        d = []
        for s in range(X.shape[1]):
            print(s)
            cmpdf = self._cmpdf(X[:,[s]], column=s)

            U = np.array([(cmpdf <= p).sum() for p in cmpdf])
            U = U / n_samples

            d.append(np.max(np.abs(U-cmpdf)))

        return(np.max(d))


    def _pdf(self, X, verbose=0):
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

    def _cmpdf(self, X, column, verbose = 0):
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

    def _rosenblatt(self, X, verbose=0):
        n_samples = X.shape[0]

        list_pi = list(itertools.permutations(np.arange(X.shape[1])))

        _lambda_product = []
        U = []

        for pi in list_pi:
            if verbose > 0:
                print(pi)
            _lambda = np.zeros((n_samples, len(pi)))
            observed_columns = []
            for id_c, c in enumerate(pi):
                observed_columns.append(c)

                if len(observed_columns) == 1:
                    _lambda[:, id_c] = self._cmpdf(X=X[:,observed_columns],
                                                   column=observed_columns[0],
                                                   verbose=verbose-1)

                else:
                    _lambda[:, id_c] = self._ccpdf(X1=X[:, [observed_columns[-1]]],
                                                   X2=X[:, observed_columns[:-1]],
                                                   column_X1=observed_columns[-1],
                                                   columns_X2=observed_columns[:-1],
                                                   verbose=verbose-1)

            U_n = np.zeros(n_samples)
            for i in range(n_samples):
                U_n[i] = np.all(_lambda <= _lambda[i, :], axis=1).sum()
            U_n /= n_samples

            _lambda_product.append(np.product(_lambda, axis=1))
            U.append(U_n)

        return (_lambda_product, U)

    def ks(self, X, verbose=0):
        l, U = self._rosenblatt(X, verbose=verbose)

        return(np.max([np.max(np.abs(U[i]-l[i])) for i in range(len(U))]))
