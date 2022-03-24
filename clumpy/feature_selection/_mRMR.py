# -*- coding: utf-8 -*-

import numpy as np
from ._feature_selector import FeatureSelector

class MRMR(FeatureSelector):
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        
        F = []
        for j in range(X.shape[1]):
            F.append(np.corrcoef(np.vstack((X[:,j], y)))[0,1])
        F = np.abs(F)
        
        
        
        return(np.arange(X.shape[1]))