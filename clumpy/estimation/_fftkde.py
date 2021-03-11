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

    def predict(self, X):
        return(self._knr.predict(X))

    def evaluate(self, grid_points=None):

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
