import numpy as np

from . import bandwidth_selection
from ._density_estimator import DensityEstimator
from tqdm import tqdm
import sparse

class UKDE(DensityEstimator):
    def __init__(self,
                 h='scott',
                 q=10,
                 low_bounded_features=[],
                 high_bounded_features=[],
                 low_bounds=[],
                 high_bounds=[],
                 preprocessing='whitening',
                 forbid_null_value=False,
                 verbose=0,
                 verbose_heading_level=1):

        super().__init__(low_bounded_features=low_bounded_features,
                         high_bounded_features=high_bounded_features,
                         low_bounds=low_bounds,
                         high_bounds=high_bounds,
                         forbid_null_value=forbid_null_value,
                         verbose=verbose,
                         verbose_heading_level=verbose_heading_level)

        self.h = h
        self.q = q
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.verbose_heading_level = verbose_heading_level

    def __repr__(self):
        if self._h is None:
            return ('UKDE(h=' + str(self.h) + ')')
        else:
            return ('UKDE(h=' + str(self._h) + ')')

    def fit(self, X, y=None):
        #preprocessing
        self._set_data(X)

        # BOUNDARIES INFORMATIONS
        self._set_boundaries()

        # BANDWIDTH SELECTION
        if type(self.h) is int or type(self.h) is float:
            self._h = float(self.h)

        elif type(self.h) is str:
            if self.h == 'scott' or self.h == 'silverman':
                self._h = 2 * bandwidth_selection.scotts_rule(X)
            else:
                raise (ValueError("Unexpected bandwidth selection method."))
        else:
            raise (TypeError("Unexpected bandwidth type."))

        if self.verbose > 0:
            print('Bandwidth selection done : h=' + str(self._h))

        # GRID FITTING
        self._bins = []
        self._digitized_data = np.zeros(self._data.shape)
        for k in range(self._d):
            self._bins.append(np.arange(self._data[:,k].min(),
                                  self._data[:,k].max()+self._h,
                                  self._h / self.q))
            self._digitized_data[:,k] = np.digitize(x=self._data[:,k],
                                                    bins=self._bins[k],
                                                    right=False)

        self._digitized_data = self._digitized_data.astype(int)

        # uniques, nb = np.unique(self._digitized_data, axis=0, return_counts=True)
        #
        # cells = {tuple(u):nb for u in uniques}
        # cells_copy = cells.copy()
        #
        # for u in tqdm(uniques, total=uniques.shape[0]):
        #     key = tuple(u + np.array([1,0,0]))
        #     if key in cells.keys():
        #         cells[tuple(u)] += cells_copy[key]
        #
        # self._cells = cells

        # self._cells = sparse.COO(coords=uniques.T,
        #                          data=nb,
        #                          shape=[bins.size+1 for bins in self._bins])


        return (self)

    def predict(self, X):
        # get indices outside bounds
        # it will be use to cut off the result later
        id_out_of_low_bounds = np.any(X[:, self.low_bounded_features] < self.low_bounds, axis=1)
        id_out_of_high_bounds = np.any(X[:, self.high_bounded_features] > self.high_bounds, axis=1)

        # X preprocessing
        if self.preprocessing != 'none':
            X = self._preprocessor.transform(X)

        # f initialization
        f = np.zeros(X.shape[0])

        # X digitization
        digitized_X = np.zeros(X.shape)
        for k in range(self._d):
            digitized_X[:,k] = np.digitize(x=X[:,k],
                                           bins=self._bins[k],
                                           right=False)

        digitized_X = digitized_X.astype(int)

        unique_X, indices = np.unique(digitized_X, axis=0, return_inverse=True)

        f_unique = np.zeros(unique_X.shape[0])

        for i, u in tqdm(enumerate(unique_X), total=unique_X.shape[0]):
            f_unique[i] = self._cells[tuple(u)]

        f = f_unique[indices]
        print(f.max())
        f /= self._n

        # outside bounds : equal to 0
        f[id_out_of_low_bounds] = 0
        f[id_out_of_high_bounds] = 0

        # Preprocessing correction
        if self.preprocessing != 'none':
            f /= np.product(self._preprocessor.scale_)

        return(f)
