# -*- coding: utf-8 -*-

import numpy as np
from ._feature_selector import FeatureSelector

class MRMR(FeatureSelector):
    def __init__(self, s):
        self.s = s
    
    def fit(self, X, y):
        
        F = []
        for j in range(X.shape[1]):
            F.append(np.abs(np.corrcoef(np.vstack((X[:,j], y)))[0,1]))
        F = np.abs(F)
        
        corr_X = np.abs(np.corrcoef(X))
        
        T_bar = np.arange(X.shape[1])
        
        id_max = np.argmax(F)
        
        T_bar = np.delete(T_bar, id_max)
        T = np.array([id_max])
        
        for j in range(self.s):
            
            alpha = F[T_bar] / np.sum(corr_X[T_bar, :][:, T], axis=1)
            
            b = np.argmax(alpha)
            T_bar = np.delete(T_bar, b)
            T = np.append(T, b)
        
        self._cols_support = list(T)
        
        return(self)