import numpy as np

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

        Parameters
        ----------
        features : list of int
            Features of the marginal probability.
        """
        if len(features) == self.X.shape[1]:
            return(self.X, self.y)

        features_bar = np.delete(np.arange(self.X.shape[1]), features)

        M_y = self.y.reshape(self.grid_shape)

        v = np.product(self._dx[features_bar])
        n = np.product(np.array(self.grid_shape)[features_bar])

        marginal = M_y.sum(axis=tuple(features_bar)) * v / n

        m = self.X.min(axis=0)
        idx = np.all(self.X[:, features_bar] == m[features_bar], axis=1)
        return(self.X[:, features][idx], marginal.flat[:])

    def cumulative(self):
        M_y = self.y.reshape(self.grid_shape)
        # cumulative = np.array([np.all(self.X <= x, axis=1).sum() for x in self.X])
        # cumulative /= self.X.shape[0]

        # return(self.X, cumulative)

        # def multidim_cumsum(a):
        out = M_y.cumsum(-1)
        for i in range(2,M_y.ndim+1):
            np.cumsum(out, axis=-i, out=out)

        out *= self._v / self.X.shape[0]

        return(out)

    def cumulative_marginal(self, features):
        x_marginal, y_marginal = self.marginal(features=features)

        M_y_marginal = y_marginal.reshape(tuple(np.array(self.grid_shape)[features]))

        out = M_y_marginal.cumsum(-1)
        for i in range(2, M_y_marginal.ndim + 1):
            np.cumsum(out, axis=-i, out=out)

        v = np.product(self._dx[features])
        n = np.product(np.array(self.grid_shape)[features])

        out *= v / n

        return(x_marginal, out)

    # def conditional(self, features_X1, features_X2):
        


