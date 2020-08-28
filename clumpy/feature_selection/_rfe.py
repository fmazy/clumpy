#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 15:48:52 2020

@author: frem
"""

from ..calibration import compute_P_vf__vi_z

class RFE():
    """
    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.
    n_features_to_select : int or None (default=None)
        The number of features to select. If `None`, half of the features
        are selected.
    step : int or float, optional (default=1)
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of features to remove at each iteration.
    verbose : int, (default=0)
        Controls verbosity of output.
    """
    def __init__(self, estimator, n_features_to_select=None, step=1, verbose=0):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose
    
    def fit(self, J):
        
        features_names = J[['z']].columns.to_list()
        
        if self.n_features_to_select is None:
            n_features_to_select = len(features_names) // 2
        else:
            n_features_to_select = self.n_features_to_select
            
        print(n_features_to_select,' features to select')
        
        while len(features_names) > n_features_to_select:
            print('remaining features : ', features_names)
            
            print('computing P_vf__vi_z')
            P_vf__vi_z = compute_P_vf__vi_z(J=J)
            
            print(P_vf__vi_z)
            
            estimator.fit(P_vf__vi_z)
            
            # return(estim1)
            
            break
        
        return(J)