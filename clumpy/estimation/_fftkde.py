from KDEpy import FFTKDE as KDEpy_FFTKDE
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

class FFTKDE(KDEpy_FFTKDE):
    def __init__(self, kernel='gaussian', bw='1', bounded_features=[]):
        super().__init__(kernel = kernel,
                         bw = bw)

        self.bounded_features = bounded_features

    def fit(self, X, weights=None):
        # first, create data mirror in case of bounded columns
        self.low_bounds = X[:, self.bounded_features].min(axis=0)

        for idx, feature in enumerate(self.bounded_features):
            X_mirrored = X.copy()
            X_mirrored[:, feature] = 2 * self.low_bounds[idx] - X[:, feature]

            X = np.vstack((X, X_mirrored))

            if weights is not None:
                weights = np.vstack((weights, weights))
        super().fit(data=X,
                    weights=weights)

    def fit_knr(self, grid_points):
        X_grid, y = self.evaluate(grid_points)

        self._knr = KNeighborsRegressor(n_neighbors=1)
        self._knr.fit(X_grid, y)

    def evaluate(self, grid_points=None):
        """
        evaluate the kde through a grid
        """
        X_grid, y = super().evaluate(grid_points=grid_points)
        if len(X_grid.shape) == 1:
            X_grid = X_grid[:,None]
        # Set the KDE to zero outside of the domain
        # samples are removed
        # y[np.any(X_grid[:,self.bounded_features] < self.low_bounds, axis=1)] = 0
        idx = np.all(X_grid[:, self.bounded_features] >= self.low_bounds, axis=1)

        y = y[idx]
        X_grid = X_grid[idx, :]

        # multiply the y-values to get integral of ~1
        y = y * 2**len(self.bounded_features)

        return(X_grid, y)

    def predict(self, X):
        p = self._knr.predict(X)

        p[np.any(X[:, self.bounded_features] < self.low_bounds, axis=1)] = 0

        return(p)

    def sample(self, n, exact=False):
        # first sample through the kernel
        kernel_sample = self._kernel_sample(n = n*2**len(self.bounded_features), exact = exact)

        # sample a mu element
        idx = np.random.choice(a = self.data.shape[0],
                               size = kernel_sample.shape[0],
                               replace = True)

        X = self.data[idx, :] + kernel_sample

        return(X[np.all(X[:, self.bounded_features] >= self.low_bounds, axis=1)])

    def _kernel_sample(self, n, exact=False):
        """
        kernel through the kernel with the rejection sampling method.
        """
        # first, get the support
        support = self._kernel_practical_support

        # get a first sample. A priori, the number of samples will be insufficient
        X = support * (2 * np.random.random((n, self.data.shape[1])) - 1)
        pdf_X = self.kernel.evaluate(X, bw=self.bw, norm=self.norm)
        C = np.random.random(n) * pdf_X.max()

        X = X[C < pdf_X, :]

        # now, estimate the needed η
        X_shape_for_n = X.shape[0]
        n_needed = int((n - X.shape[0]) * n / X.shape[0])

        # get new samples
        X_new = support * (2 * np.random.random((n_needed, self.data.shape[1])) - 1)
        pdf_X = self.kernel.evaluate(X_new, bw=self.bw, norm=self.norm)
        C = np.random.random(n_needed) * pdf_X.max()

        # X_new = X_new[C < pdf_X, :]
        X = np.vstack((X, X_new[C < pdf_X, :]))

        if exact:
            if X.shape[0] > n:
                return (X[:n, :])

            n_needed = int((n - X.shape[0]) * n / X_shape_for_n * 2)

            # get new samples
            X_new = support * (2 * np.random.random((n_needed, self.data.shape[1])) - 1)
            pdf_X = self.kernel.evaluate(X_new, bw=self.bw, norm=self.norm)
            C = np.random.random(n_needed) * pdf_X.max()

            # X_new = X_new[C < pdf_X, :]
            X = np.vstack((X, X_new[C < pdf_X, :]))

            return (X[:n, :])

        return (X)


