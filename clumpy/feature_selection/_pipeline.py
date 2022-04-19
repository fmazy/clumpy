# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy

from ._feature_selector import FeatureSelector

class Pipeline(FeatureSelector):
    def __init__(self, fs_list):
        self.list = fs_list
        
        super().__init__()
        
    def __repr__(self):
        return("Pipeline"+str(self.list))
    
    def _fit(self, X, y=None):
        
        self._cols_support = np.arange(X.shape[1])[None, :]
        
        for fs in self.list:
            X = fs.fit_transform(X, y)
            self._cols_support = fs.transform(self._cols_support)
        
        self._cols_support = self._cols_support[0]
        
        return(self)
    
    def copy(self):
        return(deepcopy(self))

    def check(self, objects=[]):
        for selector in self.list:
            if selector in objects:
                raise(ValueError("Selector objects must be different."))
            else:
                objects.append(selector)
            