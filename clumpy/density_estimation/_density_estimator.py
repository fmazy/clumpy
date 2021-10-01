#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:01:44 2021

@author: frem
"""

from sklearn.base import BaseEstimator
import numpy as np


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

    def fit(self, X, y):
        return (self)

    def predict(self, X):
        return (np.zeros(X.shape[0]))
