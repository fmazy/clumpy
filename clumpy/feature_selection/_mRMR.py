# -*- coding: utf-8 -*-

import numpy as np
from ._feature_selector import FeatureSelector

class MRMR(FeatureSelector):
    def __init__(self, s):
        self.s = s
    
    def fit(self, X, y):
        
        F = []
        for j in range(X.shape[1]):
            F.append(np.corrcoef(np.vstack((X[:,j], y)))[0,1])
        F = np.abs(F)
        
        T_bar = np.arange(X.shape[1])
        
        id_max = np.argmax(F)
        
        T_bar = np.delete(T_bar, id_max)
        T = np.array([id_max])
        
        return(np.arange(X.shape[1]))