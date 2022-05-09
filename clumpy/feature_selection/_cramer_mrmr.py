# -*- coding: utf-8 -*-

import numpy as np
from ._feature_selector import FeatureSelector

class CramerMRMR(FeatureSelector):
    def __init__(self, 
                 e=-1):
        self.e = e
        
        super().__init__()
    
    def __repr__(self):
        return 'CramerMRMR(e='+str(self.e)+')'
    
    def _fit(self, X, y):
        
        self._cols_support = []
        
        return(self)
    
    # def _relevance(self):
    
    def _bin_width(self, x, y):
        
        n, d = X.shape
        
        delta__k = np.std(X, axis=0) * 0.01
        
        bins__k = [np.arange(start=X[:,i].min(), 
                          stop=X[:,i].max(),
                          step=delta__k[i]) for i in range(d)]
        
        gamma__k = [np.digitize(x=X[:,i],
                                bins=bins__k[i]) for i in range(d)]
        
        n_gamma__k = [np.unique(gamma, return_counts=True)[1] for gamma in gamma__k]
        
        def diff(n_gamma, n):
            return np.abs(n_gamma - 2 * (n_gamma * (1 - n_gamma/n))**0.5)
        
        test__k = [np.all(diff(n_gamma,n) < self.epsilon * n_gamma) for n_gamma in n_gamma__k]
        test = np.all(test__k)