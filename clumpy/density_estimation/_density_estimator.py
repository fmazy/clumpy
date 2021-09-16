#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:01:44 2021

@author: frem
"""

from sklearn.base import BaseEstimator


class DensityEstimator(BaseEstimator):
    def __init__(self,
             low_bounded_features=[],
             high_bounded_features=[],
             low_bounds = [],
             high_bounds = [],
             forbid_null_value = False,
             verbose=0):
        self.low_bounded_features = low_bounded_features
        self.high_bounded_features = high_bounded_features
        self.low_bounds = low_bounds
        self.high_bounds = high_bounds
        self.forbid_null_value = forbid_null_value
        self.verbose = verbose
        
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


class Parameters():
    """
    Density Estimators Parameters.
    Note that boundary parameters and ``forbid_null_value`` are set
    automatically when the density estimator is called by a model class.

    Parameters
    ----------
    method : {'gkde'} or DensityEstimator
        The density estimator method or a callable class. For now, only
        the GKDE method is avaiable.
        
            gkde : Gaussian Kernel Density Estimator.
            
    **params : kwargs
        Density estimator parameters.

    """
    def __init__(self,
                method='gkde',
                **params):
        self.method = method 
        self.params = params