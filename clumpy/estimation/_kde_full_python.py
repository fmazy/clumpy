from sklearn.base import BaseEstimator
import numpy as np

class KernelDensityFullPython(BaseEstimator):
    def __init__(self,
                 H=None,
                 bandwidth_selection=None,
                 diag=True,
                 gridsize=None,
                 bounded_features=[],
                 binned=False,
                 supp_tol=3.7,
                 verbose=False):

        self.bandwidth_selection = bandwidth_selection
        self.H = H
        self.gridsize = gridsize
        self.diag = diag
        self._selected_H = None
        self.bounded_features = bounded_features
        self.binned = binned
        self.supp_tol = supp_tol
        self.verbose = verbose

    def fit(self,
            X,
            y=None,
            sample_weight=None):

        # mirror datas for bounded features
        X = X.copy()
        # first, create data mirror in case of bounded columns
        self._low_bounds = X[:, self.bounded_features].min(axis=0)

        for idx, feature in enumerate(self.bounded_features):
            X_mirrored = X.copy()
            X_mirrored[:, feature] = 2 * self._low_bounds[idx] - X[:, feature]

            X = np.vstack((X, X_mirrored))

            if sample_weight is not None:
                sample_weight = np.vstack((sample_weight, sample_weight))

        self._data = X
        if sample_weight is not None:
            self._data_weights = sample_weight
        else:
            self._data_weights = None

    def grid_density(self):
        if self.binned:

            lin1dbins = [np.linspace(self._data[:,j].min(), self._data[:,j].max(), self.gridsize[j]) for j in range(self._data.shape[1])]

            cols = np.meshgrid(*lin1dbins, indexing='ij')

            X_grid = np.vstack(tuple(col.flat for col in cols)).T

        else:
            print('non binned not yet implemented')

        ind_mat = 

        return(X_grid, 1)



