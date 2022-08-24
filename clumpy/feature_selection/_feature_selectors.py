#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 22:16:44 2021

@author: frem
"""
import numpy as np
import pandas as pd

from ..layer import FeatureLayer
from .._base import State

class FeatureSelectors():
    def __init__(self, selectors={}):
        self._fitted = False
        
        self.selectors = selectors
    
    def add_selector(self, v, selector):
        self.selectors[v] = selector
    
    def fit(self, Z, V, bounds):
        n, d = Z.shape
        list_v = np.unique(V)
        id_evs = np.zeros(d).astype(bool)
        
        for v in list_v:
            if v in self.selectors.keys():
                transited_pixels = V == v
                self.selectors[v].fit(Z=Z,
                                      transited_pixels=transited_pixels,
                                      bounds=bounds)
                
                id_evs[self.selectors[v]._cols_support] = True
        
        self._cols_support = np.arange(d)[id_evs]
        
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
    
    def get_selected_features(self, features):
        return([features[i] for i in self._cols_support])
    
    def get_bounds(self, features):
        selected_features = self.get_selected_features(features)
        
        bounds = []
        for id_col, item in enumerate(selected_features):
            if isinstance(item, FeatureLayer):
                if item.bounded in ['left', 'right', 'both']:
                    # one takes as parameter the column id of
                    # bounded features AFTER feature selection !
                    bounds.append((id_col, item.bounded))
                    
            # if it is a state distance, add a low bound set to 0.0
            if isinstance(item, State) or isinstance(item, int):
                bounds.append((id_col, 'left'))
        
        return(bounds)
    
    
        

