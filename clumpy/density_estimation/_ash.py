import numpy as np
from tqdm import tqdm
# import sparse
import pandas as pd

from ._density_estimator import DensityEstimator
from . import bandwidth_selection
from ..tools._console import title_heading

class Digitize():
    def __init__(self, dx, shift=0):
        self.dx = dx
        self.shift = shift

    def fit(self, X):
        self._d = X.shape[1]
        self._bins = [np.arange(V.min() - self.dx + self.shift,
                                V.max() + self.dx + self.shift,
                                self.dx) for V in X.T]

        return (self)

    def transform(self, X):
        X = X.copy()
        for k in range(self._d):
            X[:, k] = np.digitize(X[:, k], bins=self._bins[k])
        return (X.astype(int))

    def fit_transform(self, X):
        self.fit(X)

        return (self.transform(X))


class ASH(DensityEstimator):
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

        self.preprocessing = preprocessing
        self.h = h
        self._h = None
        self.q = q

    def __repr__(self):
        if self._h is None:
            return('ASH(h='+str(self.h)+')')
        else:
            return('ASH(h='+str(self._h)+')')

    def fit(self, X):
        # preprocessing
        self._set_data(X)

        # BOUNDARIES INFORMATIONS
        self._set_boundaries()

        # BANDWIDTH SELECTION
        if type(self.h) is int or type(self.h) is float:
            self._h = float(self.h)

        elif type(self.h) is str:
            if self.h == 'scott' or self.h == 'silverman':
                self._h = 2.576 * bandwidth_selection.scotts_rule(X)
            else:
                raise (ValueError("Unexpected bandwidth selection method."))
        else:
            raise (TypeError("Unexpected bandwidth type."))

        if self.verbose > 0:
            print('Bandwidth selection done : h=' + str(self._h))

        # NORMALIZATION FACTOR
        self._normalization = 1 / (self._h ** self._d)

        # create a digitization for each shift
        self._digitizers = []
        self._histograms = []
        for i_shift in tqdm(range(self.q)):
            self._digitizers.append(Digitize(dx=self._h,
                                             shift=self._h / self.q * i_shift))
            X_digitized = self._digitizers[i_shift].fit_transform(self._data)

            df = pd.DataFrame(X_digitized)
            df_uniques = df.groupby(by=df.columns.to_list()).size().reset_index(name='P')
            df_uniques['P'] /= self._n

            self._histograms.append(df_uniques)

        return(self)

    def predict(self, X):
        # get indices outside bounds
        # it will be use to cut off the result later
        id_out_of_low_bounds = np.any(X[:, self.low_bounded_features] < self.low_bounds, axis=1)
        id_out_of_high_bounds = np.any(X[:, self.high_bounded_features] > self.high_bounds, axis=1)

        if self.preprocessing != 'none':
            X = self._preprocessor.transform(X)

        f = np.zeros(X.shape[0])

        for i_shift in tqdm(range(self.q)):
            X_digitized = self._digitizers[i_shift].transform(X)

            df = pd.DataFrame(X_digitized)
            df = df.merge(self._histograms[i_shift], how='left')
            df.fillna(value=0.0, inplace=True)

            f += df.P.values

        # Normalization
        f *= self._normalization / self.q

        # boundary bias correction
        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level) + 'Boundary bias correction...')

        f /= self._boundary_correction(X, self._h)

        # outside bounds : equal to 0
        f[id_out_of_low_bounds] = 0
        f[id_out_of_high_bounds] = 0

        # Preprocessing correction
        if self.preprocessing != 'none':
            f /= np.product(self._preprocessor.scale_)

        # if null value is forbiden
        if self.forbid_null_value or self._force_forbid_null_value:
            if self.verbose > 0:
                print(title_heading(self.verbose_heading_level) + 'Null value correction...')
            idx = f == 0.0

            m_0 = idx.sum()

            new_n = self._n + m_0

            f = f * self._n / new_n

            min_value = 1 / new_n * self._normalization * 1
            f[f == 0.0] = min_value

            # Warning flag
            # check the relative number of corrected probabilities
            if self.verbose > 0:
                print('m_0 = ' + str(m_0) + ', m = ' + str(self._n) + ', m_0 / m = ' + str(
                    np.round(m_0 / self._n, 4)))

            # warning flag
            if m_0 / self._n > 0.01:
                print('WARNING : m_0/m > 0.01. The parameter `n_fit_max` should be higher.')

            if self.verbose > 0:
                print('Null value correction done for ' + str(m_0) + ' elements.')

        return (f)

    def _boundary_correction(self, X, h):
        """
        X in the WT space.
        """
        correction = np.ones(X.shape[0])
        for hyperplane in self._low_bounds_hyperplanes + self._high_bounds_hyperplanes:
            dist = hyperplane.distance(X, p=np.inf)
            correction *= 1 - np.maximum(0, dist + h) * ( 1 - dist / h) / 2

        return(correction)
