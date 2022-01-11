import numpy as np

from . import bandwidth_selection
from ._density_estimator import DensityEstimator
from tqdm import tqdm
# from ..tools._data import Digitize
from KDEpy import FFTKDE as FFTKDE_KDEpy

class FFTKDE(DensityEstimator):
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
            return ('FFTKDE(h=' + str(self.h) + ')')
        else:
            return ('FFTKDE(h=' + str(self._h) + ')')

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
                self._h = bandwidth_selection.scotts_rule(X)
            else:
                raise (ValueError("Unexpected bandwidth selection method."))
        else:
            raise (TypeError("Unexpected bandwidth type."))

        if self.verbose > 0:
            print('Bandwidth selection done : h=' + str(self._h))

        # digitize
        digitize = Digitize(self._h / self.q)
        digitized_data = digitize.fit_transform(self._data)

        uniques, nb = np.unique(digitized_data, return_counts=True, axis=0)
        
        # grid
        grid_points = tuple([int((self._data[:,k].max() - self._data[:,k].min()) / (self._h)) for k in range(self._d)])
        print(grid_points, np.product(grid_points))
        # self._x, self._f = FFTKDE_KDEpy(bw=self._h).fit(self._data).evaluate(grid_points)
        #
        # grid_points = tuple([bins.size for bins in digitize._bins])
