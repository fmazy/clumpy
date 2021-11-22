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

class DensityEstimator(BaseEstimator):
    def __init__(self,
                 low_bounded_features=[],
                 high_bounded_features=[],
                 low_bounds=[],
                 high_bounds=[],
                 forbid_null_value=False,
                 verbose=0,
                 verbose_heading_level=1):
        self.low_bounded_features = low_bounded_features
        self.high_bounded_features = high_bounded_features
        self.low_bounds = low_bounds
        self.high_bounds = high_bounds
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
        # low bounds
        if self.low_bounds is None or len(self.low_bounds) != len(self.low_bounded_features):
            raise (ValueError("unexpected low bounds value"))

        self._low_bounds_hyperplanes = []
        for id_k, k in enumerate(self.low_bounded_features):
            A = np.diag(np.ones(self._d))
            A[:, k] = self.low_bounds[id_k]

            A_wt = self._preprocessor.transform(A)

            self._low_bounds_hyperplanes.append(Hyperplane().set_by_points(A_wt))

        # high bounds
        if self.high_bounds is None or len(self.high_bounds) != len(self.high_bounded_features):
            raise (ValueError("unexpected low bounds value"))

        self._high_bounds_hyperplanes = []
        for id_k, k in enumerate(self.high_bounded_features):
            A = np.diag(np.ones(self._d))
            A[:, k] = self.high_bounds[id_k]

            A_wt = self._preprocessor.transform(A)

            self._high_bounds_hyperplanes.append(Hyperplane().set_by_points(A_wt))

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
