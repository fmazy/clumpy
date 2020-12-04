#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 11:11:41 2020

@author: frem
"""

import numpy as np

from ._calibration import _Calibration

class ClusterStraight(_Calibration):
    def _new_estimator(self):
        return(_ClusterStraightEstimator())
    
class _ClusterStraightEstimator():
    def fit(self, X, y):
        self.y = np.zeros((X.max()+1, y.shape[1]))
        self.y[X.reshape(-1)] = y
        
    def predict(self, X):
        
        # check if all X are represented
        id_s_not_represented = np.isin(element=np.arange(self.y.shape[0]),
                                       test_elements=np.unique(X.reshape(-1)),
                                       assume_unique=True,
                                       invert=True)
        
        # if not, edit y to have sum(y) = 1 along all unique X
        s = self.y[id_s_not_represented,:].sum(axis=0)
        
        
        return(self.y[X.reshape(-1),:]/(1-s))