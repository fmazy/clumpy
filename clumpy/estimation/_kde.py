#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:15:42 2021

@author: frem
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor
from KDEpy import FFTKDE, NaiveKDE, TreeKDE
from numpy import linalg

class KDE(BaseEstimator):
    def __init__(self,
                 bw=None,
                 grid_points=None,
                 grid_dx=None,
                 bounded_features=[],
                 method='FFTKDE',
                 verbose=0):
    
        self.bw = bw
        self.grid_points = grid_points
        self.grid_dx = grid_dx
        self.bounded_features = bounded_features
        self.method= method
        self.verbose = verbose
        
    def fit(self, X, y=None):
        self._num_obs = X.shape[0]
        self._num_dims = X.shape[1]
        
        # we now perform singular value decomposition of X
        # see https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
        self._U, self._s, Vt = linalg.svd(X, full_matrices=False)
        self._V = Vt.T
        
        # transform the data to principal component space
        transformed_X = self._transform_to_principal_component_space(X)
        
        if type(self.bw) is str:
            if self._bw == 'UCV':
                self._bw = self._compute_bw_ucv(X)
            else:
                raise(ValueError('Unexpected bandwidth selection method'))
        elif type(self.bw) is float:
            self._bw = self.bw
        else:
            raise(TypeError('Unexpected bandwidth type : it should be a method among {\'UCV\'} or a float scalar.'))
        
        if self.method == 'FFTKDE':
            KDE_class = FFTKDE
        elif self.method == 'TreeKDE':
            KDE_class = TreeKDE
        elif self.method == 'NaiveKDE':
            KDE_class = NaiveKDE
        else:
            raise(ValueError("Unexpected method argument. Should be {'NaiveKDE, TreeKDE, FFTKDE'}"))
        
        # --------- COMPUTE KERNEL DENSITY ESTIMATION ON ROTATED DATA ---------
        
        # Compute the kernel density estimate
        
        kde = KDE_class(kernel='gaussian', norm=2, bw=self._bw)
        grid, points = kde.fit(transformed_X).evaluate(self.grid_points)
        
        # --------- ROTATE THE GRID BACK ---------
        # After rotation, the grid will not be alix-aligned
        grid_rot = grid / np.sqrt(self._num_obs) @ np.linalg.inv(self._V @ np.diag(1/self._s))
        
        # --------- RESAMPLE THE GRID ---------
        # We pretend the data is the rotated grid, and the f(x, y) values are weights
        # This is a re-sampling of the KDE onto an axis-aligned grid, and is needed
        # since matplotlib requires axis-aligned grid for plotting.
        # (The problem of plotting on an arbitrary grid is similar to density estimation)
        kde = KDE_class(kernel='gaussian', norm=2, bw=self._bw)
        self._X_grid, self._grid_density = kde.fit(grid_rot, weights=points).evaluate(self.grid_points)
        
        self._knr = KNeighborsRegressor(n_neighbors=4, weights='distance')
        self._knr.fit(grid, points)
        
    def predict(self, X):
        return(self._knr.predict(X))
        
    def _transform_to_principal_component_space(self, X):
        return(X @ self._V @ np.diag(1 / self._s) * np.sqrt(self._num_obs))
    
    def _inverse_transform_to_principal_component_space(self, X):
        return(X / np.sqrt(self._num_obs) @ np.diag(self._s)  @ self._V.T )
    
    