#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 22:16:44 2021

@author: frem
"""
import numpy as np
import pandas as pd

class FeatureSelector():
    def __init__(self):
        self._fitted = False
    
    def fit(self, X, y=None):
        self._fit(X, y)
        
        self._fitted = True
        self._n_cols = X.shape[1]
        
        return(self)
    
    def get_support(self):
        """
        Get the columns indexes that are kept for the transformation.
        """
        return(self._cols_support)
    
    def transform(self, X):
        """
        Reduce X to the selected features.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_r : array-like of shape (n_samples, n_features)
            The input samples with only the selected features.
        """
        if not self._fitted:
            raise(TypeError("The FeatureSelector object has to be fitted before calling transform()."))
        
        if X.shape[1] != self._n_cols:
            raise(ValueError("X's number of columns is incorrect. Expected "+str(self._n_cols)+", got "+str(X.shape[1])))
        
        return(X[:, self._cols_support])

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
    
    
        

