# -*- coding: utf-8 -*-

import numpy as np
from ._feature_selector import FeatureSelector
from sklearn.feature_selection import f_classif

class MRMR(FeatureSelector):
    def __init__(self, e=-1):
        self.e = e
        
        super().__init__()
        
    
    def __repr__(self):
        return 'mRMR(e='+str(self.e)+')'
    
    def _fit(self, X, y):
        n, d = X.shape
        
        nb_ev = self.e
        if nb_ev == -1 or nb_ev > d:
            nb_ev = d
        
        F = f_classif(X, y)[0]
        
        R = np.abs(np.corrcoef(X.T))
        
        T = [np.argmax(F)]
        T_bar = np.delete(np.arange(d), T[0])
        
        for j in range(nb_ev - 1):
            alpha = F[T_bar] / R[T, :][:, T_bar].sum(axis=0)
            id_k_star = np.argmax(alpha)
            T = np.append(T, T_bar[id_k_star])
            T_bar = np.delete(T_bar, id_k_star)
        
        self._cols_support = T
                
        return(self)
    
    def copy(self):
        return(MRMR(e=self.e))