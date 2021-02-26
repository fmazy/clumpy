import numpy as np
import itertools
import multiprocessing as mp

class BaseEstimator():
    def predict(self, X):
        """
        Predict the probability density function according to the training.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.

        Returns
        -------
        f : array-like of shape (n_samples)
            The estimated probability density function.
        """
        return(self._pdf(X))

    def score(self, X, verbose=0):
        """
        Computes the score which corresponds to the
        Kolmogorov-Smirnov Statistic according to Justel (1997).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.
        verbose : int, default=0
            Verbosity level.

        Returns
        -------
        s : float
            The score. The weaker it is, the better it is.
        """
        return(self.ks(X, verbose=verbose))

    def ks(self, X, n_jobs=1, verbose=0):
        """
        Computes the Kolmogorov-Smirnov Statistic according to Justel (1997).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.
        verbose : int, default=0
            Verbosity level.

        Returns
        -------
        d : float
            The KS distance. The weaker it is, the better it is.
        """
        l, U = self._justel(X, n_jobs=n_jobs, verbose=verbose)

        return(np.max([np.max(np.abs(U[i]-l[i])) for i in range(len(U))]))

    def marginal_ks(self, X, verbose=0):
        """
        Computes a simplified Kolmogorov-Smirnov statistic
        based on marginal probabilities only.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.
        verbose : int, default=0
            Verbosity level.

        Returns
        -------
        d : float
            The marginal distance. The weaker it is, the better it is.
        """
        n_samples = X.shape[0]

        d = []
        for s in range(X.shape[1]):
            if verbose>0:
                print(s)
            cmpdf = self._cmpdf(X[:,[s]], column=s)

            U = np.array([np.all(cmpdf <= p,axis=1).sum() for p in cmpdf])
            U = U / n_samples

            d.append(np.max(np.abs(U-cmpdf)))

        return(np.max(d))

    def _justel(self, X, n_jobs=1, verbose=0):
        """
        Computes the Justel's elements lambda and U
        """
        n_samples = X.shape[0]

        list_pi = list(itertools.permutations(np.arange(X.shape[1])))

        _lambda_product = []
        U = []

        for pi in list_pi:
            if verbose > 0:
                print(pi)

            _lambda = self._rosenblatt_transformation(X, pi, verbose=verbose)

            U_n = np.zeros(n_samples)
            for i in range(n_samples):
                U_n[i] = np.all(_lambda <= _lambda[i, :], axis=1).sum()
            U_n /= n_samples

            _lambda_product.append(np.product(_lambda, axis=1))

            print(U_n.min(), U_n.max())
            print(_lambda_product[-1].min(), _lambda_product[-1].max())

            U.append(U_n)

        return(_lambda_product, U)

    def _rosenblatt_transformation(self, X, pi, verbose=0):
        """
        Computes the rosenblatt transformation
        according to a combination of columns pi.
        """
        _lambda = np.zeros_like(X)
        observed_columns = []
        for id_c, c in enumerate(pi):
            observed_columns.append(c)

            if len(observed_columns) == 1:
                _lambda[:, id_c] = self._cmpdf(X=X[:, observed_columns],
                                               column=observed_columns[0],
                                               verbose=verbose - 1)

            else:
                _lambda[:, id_c] = self._ccpdf(X1=X[:, [observed_columns[-1]]],
                                               X2=X[:, observed_columns[:-1]],
                                               column_X1=observed_columns[-1],
                                               columns_X2=observed_columns[:-1],
                                               verbose=verbose - 1)

        return(_lambda)



    def _pdf(self,X, verbose=0):
        raise (ValueError("The _pdf function is expected."))

    def _cmpdf(self, X, column, verbose=0):
        raise (ValueError("The _cmpdf function is expected."))

    def _ccpdf(self, X1, X2, column_X1, columns_X2, verbose=0):
        raise (ValueError("The _ccpdf function is expected."))
