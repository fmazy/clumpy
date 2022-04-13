# -*- coding: utf-8 -*-

import numpy as np
from ._feature_selector import FeatureSelector
from sklearn.feature_selection import f_classif

class MRMR(FeatureSelector):
    def __init__(self, e):
        self.e = e
    
    def fit(self, X, V):
        
        n, d = X.shape
        
        
        F = f_classif(X, V)[0]
        
        R = np.abs(np.corrcoef(X.T))
        
        T = [np.argmax(F)]
        T_bar = np.delete(np.arange(d), T[0])
        
        for j in range(self.e - 1):
            alpha = F[T_bar] / R[T, :][:, T_bar].sum(axis=0)
            
            id_k_star = np.argmax(alpha)
            T = np.append(T, T_bar[id_k_star])
            T_bar = np.delete(T_bar, id_k_star)
        
        self._cols_support = T
        
        return(self)