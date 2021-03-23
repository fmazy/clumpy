import numpy as np
# from pathos.multiprocessing import ProcessingPool as Pool
import itertools
from tqdm import tqdm
from sklearn.neighbors import KNeighborsRegressor

class Probabilities():
    def __init__(self, X, y):
        """
        Parameters
        ----------
        X : numpy-array of shape (n_samples, n_features)
            Grid points representing the whole volume.
        y : numpy-array of shape (n_samples,)
            Probabilities.
        """
        self.X = X
        self.y = y
        self._dx = X.max(axis=0) - X.min(axis=0)
        self._v = np.product(self._dx)

        self.grid_shape = ()
        for i in range(self.X.shape[1]):
            # c'est très moche...
            self.grid_shape += (np.unique(self.X[:,i]).size,)

    def integral(self):
        """
        Computes the integral on the whole volume.

        Returns
        -------
        I : float
            The integral value.
        """
        I = self.y.sum() * self._v / self.y.size
        return(I)

    def cut(self, features, x, method='nearest', keep_all_features=False):
        """
        Get a cut through dimensions.

        Parameters
        ----------
        features : list of int
            The list of features which are set to x.

        x : list of float
            The list of values according to features.

        method : {None, 'nearest'}, default='nearest'
            Method to get the cutting values

            None
                The x value correspond to self.X values.
            nearest
                The nearest self.X values are selected.

        keep_all_features : bool, default=False
            If True, returns values for all features, including set ones.

        Returns
        -------
        x_cut : numpy-array
            Cutted grid points.

        y_cut : numpy-array
            Cutted probabilities.
        """
        if method is not None:
            m = self.X.min(axis=0)
            for i, feature in enumerate(features):
                feature_bar = np.delete(np.arange(self.X.shape[1]), [feature])
                idx = np.all(self.X[:, feature_bar] == m[feature_bar], axis=1)
                u = self.X[idx, feature]
                if x[i] not in u:
                    x[i] = u[np.argmin(np.abs(u - x[i]))]

        idx = np.all(self.X[:, features] == x, axis=1)

        features_bar = np.delete(np.arange(self.X.shape[1]), features)

        if keep_all_features:
            X_return = np.zeros((idx.sum(), self.X.shape[1]))
            X_return[:, features] = x
            X_return[:, features_bar] = self.X[:, features_bar][idx]

            return(X_return, self.y[idx])
        else:
            return(self.X[:, features_bar][idx], self.y[idx])

    def marginal(self, features):
        """
        Get the marginal probabilities according to some features.
        Warning, returns X with sorted features
        Parameters
        ----------
        features : list of int
            Features of the marginal probability.
        """
        features = np.sort(features)
        if len(features) == self.X.shape[1]:
            return(self.X[:, features], self.y)

        features_bar = np.delete(np.arange(self.X.shape[1]), features)

        M_y = self.y.reshape(self.grid_shape).T

        v = np.product(self._dx[features_bar])
        n = np.product(np.array(self.grid_shape)[features_bar])

        marginal = M_y.sum(axis=tuple(features_bar)) * v / n

        m = self.X.min(axis=0)
        idx = np.all(self.X[:, features_bar] == m[features_bar], axis=1)
        return(self.X[:, features][idx], marginal.flat[:])

    def cumulative(self):
        M_y = self.y.reshape(self.grid_shape)

        out = M_y.cumsum(-1)
        for i in range(2,M_y.ndim+1):
            np.cumsum(out, axis=-i, out=out)

        out *= self._v / self.X.shape[0]

        return(self.X, out.flat[:])

    def cumulative_marginal(self, features):
        x_marginal, y_marginal = self.marginal(features=features)

        M_y_marginal = y_marginal.reshape(tuple(np.array(self.grid_shape)[features]))

        out = M_y_marginal.cumsum(-1)
        for i in range(2, M_y_marginal.ndim + 1):
            np.cumsum(out, axis=-i, out=out)

        v = np.product(self._dx[features])
        n = np.product(np.array(self.grid_shape)[features])

        out *= v / n

        return(x_marginal, out.flat[:])

    def conditional(self, features_X1, features_X2):
        features_X1_X2 = np.sort(features_X1 + features_X2)

        x1_x2, f_x1_x2 = self.marginal(features_X1_X2)
        # warning, x1_x2 is column sorted according to features index

        M_f_x1_x2 = f_x1_x2.reshape(np.array(self.grid_shape)[features_X1_X2])

        x_2, f_x2 = self.marginal(features_X2)
        # same here, x_2 is column sorted

        grid_shape_x2 = np.array(self.grid_shape)[features_X1_X2]
        for i in range(len(grid_shape_x2)):
            if features_X1_X2[i] not in features_X2:
                grid_shape_x2[i] = 1

        M_f_x2 = f_x2.reshape(grid_shape_x2)

        M = M_f_x1_x2 / M_f_x2

        return(x1_x2, M.flat[:])

    def cumulative_conditional(self, features_X1, features_X2):
        x1_x2, y_conditional = self.conditional(features_X1, features_X2)
        # warning, x1_x2 is sorted columns

        features_X1_X2 = np.sort(features_X1 + features_X2)
        M_conditional = y_conditional.reshape(np.array(self.grid_shape)[features_X1_X2])

        id_features_X1 = []
        for i, feature in enumerate(features_X1_X2):
            if feature in features_X1:
                id_features_X1.append(i)
        # print(id_features_X1)

        out = M_conditional.cumsum(-id_features_X1[0])
        for feature in id_features_X1[1:]:
            np.cumsum(out, axis=-feature, out=out)

        v = np.product(self._dx[features_X1])
        n = np.product(np.array(self.grid_shape)[features_X1])

        out *= v / n

        return(x1_x2, out.flat[:])

    def ks(self, X):
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
        l, U = self._justel(X)

        return(np.max([np.max(np.abs(U[i]-l[i])) for i in range(len(U))]))

    def _justel(self, X, verbose=0):
        """
        Computes the Justel's elements lambda and U
        """

        list_pi = list(itertools.permutations(np.arange(self.X.shape[1])))

        _lambda_product = []
        U = []

        for pi in list_pi:
            if verbose>0:
                print('permutation : ', pi)
            l, u = self._justel_computation(X, pi)
            _lambda_product.append(l)
            U.append(u)

        return(_lambda_product, U)

    def _justel_computation(self, X, pi):

        _lambda = self._rosenblatt_transformation(X, pi)

        U_n = np.zeros(_lambda.shape[0])
        for i in range(_lambda.shape[0]):
            U_n[i] = np.all(_lambda <= _lambda[i, :], axis=1).sum()
        U_n /= _lambda.shape[0]

        return(np.product(_lambda, axis=1), U_n)

    def _rosenblatt_transformation(self, X, pi):
        """
        Computes the rosenblatt transformation
        according to a combination of columns pi.
        """
        _lambda = np.zeros_like(X)
        observed_columns = []
        for id_c, c in enumerate(pi):
            observed_columns.append(c)

            if len(observed_columns) == 1:
                X_grid, F = self.cumulative_marginal(observed_columns)
                # features = observed_columns

                # print('lambda_0, cumulative marginal : ', observed_columns)

            else:
                X_grid, F = self.cumulative_conditional([observed_columns[-1]],
                                                         observed_columns[:-1])
                # features = [observed_columns[-1]] + observed_columns[:-1]

                # print('lambda_'+str(id_c)+', cumulative conditional X1=', [observed_columns[-1]], ' X2=', observed_columns[:-1])

            # print('F_max : ', F.max())
            features = np.sort(observed_columns)
            # print('features : ', features)

            knr = KNeighborsRegressor(n_neighbors=1, weights='uniform')

            knr.fit(X_grid, F)

            _lambda[:, id_c] = knr.predict(X[:, features])

        # print(_lambda.max(axis=0))

        return(_lambda)



