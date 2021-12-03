#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:01:44 2021

@author: frem
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

from ._whitening_transformer import _WhiteningTransformer
from ..utils._hyperplane import Hyperplane
from ..tools._console import title_heading

class DensityEstimator(BaseEstimator):
    def __init__(self,
                 bounds = [],
                 low_bounded_features=[],
                 high_bounded_features=[],
                 low_bounds=[],
                 high_bounds=[],
                 forbid_null_value=False,
                 verbose=0,
                 verbose_heading_level=1):

        self.bounds = bounds
        if len(self.bounds) == 0:
            self.bounds = [(low_bounded_features[i], low_bounds[i]) for i in range(len(low_bounded_features))]
            self.bounds += [(high_bounded_features[i], high_bounds[i]) for i in range(len(high_bounded_features))]

        self.forbid_null_value = forbid_null_value
        self.verbose = verbose
        self.verbose_heading_level = verbose_heading_level

        self._force_forbid_null_value = False

    def _set_data(self, X):
        # preprocessing
        if self.preprocessing == 'standard':
            self._preprocessor = StandardScaler()
            self._data = self._preprocessor.fit_transform(X)

        elif self.preprocessing == 'whitening':
            self._preprocessor = _WhiteningTransformer()
            self._data = self._preprocessor.fit_transform(X)

        else:
            self._data = X

        # get data dimensions
        self._n = self._data.shape[0]
        self._d = self._data.shape[1]

    def _set_boundaries(self):
        self._bounds_hyperplanes = []
        self._low_bound_trigger = []
        P_wt = self._data[0]
        P = self._preprocessor.inverse_transform(P_wt[None,:])[0]

        for k, value in self.bounds:
            A = np.diag(np.ones(self._d))
            A[:, k] = value

            A_wt = self._preprocessor.transform(A)

            hyp = Hyperplane().set_by_points(A_wt)
            hyp.set_positive_side(P_wt)
            self._bounds_hyperplanes.append(hyp)

            if P[k] >= value:
                self._low_bound_trigger.append(True)
            else:
                self._low_bound_trigger.append(False)

    def _forbid_null_values_process(self, f):
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

        return(f)

    def set_params(self, **params):
        """
        Set parameters.

        Parameters
        ----------
        **params : kwargs
            Parameters et values to set.

        Returns
        -------
        self : DensityEstimator
            The self object.

        """
        for param, value in params.items():
            setattr(self, param, value)

class NullEstimator(DensityEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return (self)

    def predict(self, X):
        return (np.zeros(X.shape[0]))
