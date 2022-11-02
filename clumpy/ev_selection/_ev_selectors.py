 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 22:16:44 2021

@author: frem
"""
import numpy as np
import pandas as pd

from ..layer import EVLayer
from .._base import State

class EVSelectors():
    def __init__(self, selectors=[]):
        self._fitted = False
        
        self.selectors = selectors
    
    def add_selector(self, selector):
        if selector not in self.selectors:
            self.selectors.append(selector)
        
        return self
    
    def fit(self, W, Z, bounds):
        print('EV selectors fit...')
        n, d = Z.shape
        # print('observed final states : ', list_v)
        id_evs = np.zeros(d).astype(bool)
        
        for i, selector in enumerate(self.selectors):
            if W[:,i].sum() > 0:
                selector.fit(Z=Z,
                             w=W[:,i],
                             bounds=bounds)
                
                id_evs[selector._cols_support] = True
        
        self._cols_support = np.arange(d)[id_evs]
        
        print('keep', self._cols_support)
        
        self._fitted = True
        self._n_cols = Z.shape[1]
        
        return(self)
    
    def get_support(self):
        """
        Get the columns indexes that are kept for the transformation.
        """
        return(self._cols_support)
    
    def transform(self, Z):
        """
        Reduce Z to the selected features.

        Parameters
        ----------

        Z : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        Z_r : array-like of shape (n_samples, n_features)
            The input samples with only the selected features.
        """
        if not self._fitted:
            raise(TypeError("The FeatureSelector object has to be fitted before calling transform()."))
            
        
        if type(Z) is list:    
            Z = np.array(Z)
            
        inline = False
        if len(Z.shape) == 1:
            Z = Z[None,:]
            inline=True
        
        if Z.shape[1] != self._n_cols:
            raise(ValueError("Z's number of columns is incorrect. EZpected "+str(self._n_cols)+", got "+str(Z.shape[1])))
        
        if inline:
            return(Z[:, self._cols_support][0])
        else:
            return(Z[:, self._cols_support])

    def fit_transform(self, X, y):
        """
        Fit to data, then transform it.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Sample vectors from which to fit and transform.

        Returns
        -------
        X_r : array-like of shape (n_samples, n_features)
            The input samples with only the selected features.
        """
        self.fit(X, y)
        return(self.transform(X))
    
    def get_selected_evs(self, evs):
        return([evs[i] for i in self._cols_support])

