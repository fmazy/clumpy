#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 09:24:01 2021

@author: frem
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import gamma

class RectKDE():
    def __init__(self,
                 h,
                 p=2,
                 bounded_features=[],
                 htol=1e-3,
                 Jtol=1e-2,
                 verbose=0):
        self.h = h
        self.p = p
        self.bounded_features=bounded_features
        self.htol = htol
        self.Jtol = Jtol
        self.verbose = verbose
    
    def fit(self, X):
        
        if self.verbose > 0:
            print('input data shape : ', X.shape)
            
            if len(self.bounded_features) > 0:
                print('\nsymetry for bounds')
                print('bounded_features : ', self.bounded_features)
        
        
        
        X = self._mirror(X)
        
        self._whitening_transformer = _WhiteningTransformer()
        X = self._whitening_transformer.fit_transform(X)
        
        X = self._cut_mirror(X)
        
        self._data = X
        
        self._n = self._data.shape[0]
        self._d = self._data.shape[1]
        
        self._nn = NearestNeighbors(radius=self.h, p=self.p)
        self._nn.fit(self._data)
        
        # mirror coefficient ?
        if len(self.bounded_features) > 0:
            grid_points = np.array([2000, 1000])
            
            self._mirror_coef = None
            X_grid, pred_grid, integral = self.grid_predict(grid_points, return_integral=True)
            
            print(integral)
            
        return(self)
    
    def predict(self, X):
        X_trans = self._whitening_transformer.transform(X)
        
        neigh_ind = self._nn.radius_neighbors(X_trans, return_distance=False)
        volume = volume_unit_ball(d=self._d, p=self.p)
        
        density = np.array([ni.size for ni in neigh_ind])
        density = density / ( self._n * self.h**self._d * volume )
        
        if self._mirror_coef is not None:
            density[np.any(X[:, self.bounded_features] < self._low_bounds, axis=1)] = 0
        
        return(density)
    
    def grid_predict(self, grid_points, return_integral=False):
        support_min, support_max = self._support_in_transformed_space()
        
        xk = np.meshgrid(*(np.linspace(support_min[k], support_max[k], grid_points[k]) for k in range(self._d)))
        X_grid = np.vstack([xk[k].flat for k in range(self._d)]).T
        
        X_grid = self._whitening_transformer.inverse_transform(X_grid)
        
        pred_grid = self.predict(X_grid)
        
        integral = pred_grid.sum() * np.product(support_max - support_min) / pred_grid.size
        
        print(integral)
        
        if return_integral:
            return(X_grid, pred_grid, integral)
        else:
            return(X_grid, pred_grid)
    
    def _mirror(self, X):
        self._low_bounds = X[:, self.bounded_features].min(axis=0)
        
        for idx, feature in enumerate(self.bounded_features):
            X_mirrored = X.copy()
            X_mirrored[:, feature] = 2 * self._low_bounds[idx] - X[:, feature]

            X = np.vstack((X, X_mirrored))
        
        if self.verbose > 0 and len(self.bounded_features) > 0:
            print('mirrored data shape : ', X.shape)
            
        return(X)
    
    def _cut_mirror(self, X):
        
        X_inv_trans = self._whitening_transformer.inverse_transform(X)
        
        X = X[np.all(X_inv_trans[:, self.bounded_features] >= self._low_bounds - self.h / 2, axis=1)]
        
        if self.verbose>0:
            print('cutted mirror data shape : ', X.shape)
        
        return(X)
    
    def _support_in_transformed_space(self):
        support_min = self._data.min(axis=0) - self.h/2
        support_max = self._data.max(axis=0) + self.h/2
        
        return(support_min, support_max)

class _WhiteningTransformer():
    def fit(self, X):
        self._mean = X.mean(axis=0)
        
        self._num_obs = X.shape[0]
        
        self._U, self._s, Vt = np.linalg.svd(X - self._mean, full_matrices=False)
        self._V = Vt.T
        
        return(self)
        
    def transform(self, X):
        X = X - self._mean
        return(X @ self._V @ np.diag(1 / self._s) * np.sqrt(self._num_obs-1))
    
    def inverse_transform(self, X):
        X = X / np.sqrt(self._num_obs-1) @ np.diag(self._s)  @ self._V.T
        X = X + self._mean
        return(X)
    
    def fit_transform(self, X):
        self.fit(X)
        
        return(self.transform(X))
    
def volume_unit_ball(d, p=2):
    """
    Volume of d-dimensional unit ball under the p-norm. When p=1 this is called
    a cross-polytype, when p=2 it's called a hypersphere, and when p=infty it's
    called a hypercube.

    Notes
    -----
    See the following paper for a very general result related to this:

    - Wang, Xianfu. “Volumes of Generalized Unit Balls.”
      Mathematics Magazine 78, no. 5 (2005): 390–95.
      https://doi.org/10.2307/30044198.
    """
    return 2.0 ** d * gamma(1 + 1 / p) ** d / gamma(1 + d / p)