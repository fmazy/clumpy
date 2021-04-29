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
from tqdm import tqdm
from KDEpy.bw_selection import silvermans_rule
from scipy import optimize
from KDEpy.kernel_funcs import _kernel_functions, p_norm
from matplotlib import pyplot as plt
import pandas as pd
from itertools import combinations
from KDEpy.kernel_funcs import volume_unit_ball

from ._box_kde import BoxKDE

_kde_class = {
    'FFTKDE':FFTKDE,
    'NaiveKDE':NaiveKDE,
    'TreeKDE':TreeKDE,
    'BoxKDE':BoxKDE}

class KDE(BaseEstimator):
    def __init__(self,
                 bw=None,
                 grid_points=None,
                 grid_dx=None,
                 bounded_features=[],
                 method='FFTKDE',
                 kernel='gaussian',
                 norm=2,
                 bw_tol=1e-2,
                 J_tol=1e-2,
                 verbose=0):
    
        self.bw = bw
        self.grid_points = grid_points
        self.grid_dx = grid_dx
        self.bounded_features = bounded_features
        self.method = method
        self.kernel = kernel
        self.norm = norm
        self.bw_tol = bw_tol
        self.J_tol = J_tol
        self.verbose = verbose
        
    def fit(self, X, y=None, compute_grid=False):
        if len(X.shape) == 1:
            X = X[:,None]
        
        if self.verbose > 0:
            print('input data shape : ', X.shape)
            
            if len(self.bounded_features) > 0:
                print('\n symetry for bounds')
                print('bounded_features : ', self.bounded_features)
            
        X = self._mirror(X)
        
        # computing grid_points according to grid_dx
        if self.grid_dx is not None:
            if self.verbose > 0:
                print('\n grid_points computing according to grid_dx')
                print('grid_points : ', self.grid_points)
            
            self.grid_points = ((X.max(axis=0) - X.min(axis=0)) / self.grid_dx).astype(int)
            self.grid_points = tuple(self.grid_points)
            
            if self.verbose > 0:
                print('grid_points : ', self.grid_points)
        
        self._num_obs = X.shape[0]
        self._num_dims = X.shape[1]
        
        # we now perform singular value decomposition of X
        # see https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
        self._U, self._s, Vt = linalg.svd(X, full_matrices=False)
        self._V = Vt.T
        
        transformed_X = self._transform_to_principal_component_space(X)
        
        if type(self.bw) is str:
            if self.bw == 'UCV':
                self._bw = self._compute_bw_ucv(transformed_X)
            else:
                raise(ValueError('Unexpected bandwidth selection method'))
        elif type(self.bw) is float:
            self._bw = self.bw
        else:
            raise(TypeError('Unexpected bandwidth type : it should be a method among {\'UCV\'} or a float scalar.'))
        
        # --------- COMPUTE KERNEL DENSITY ESTIMATION ON ROTATED DATA ---------
        # Compute the kernel density estimate
        kde = _kde_class[self.method](kernel=self.kernel, bw=self._bw, norm=self.norm)
        grid, points = kde.fit(transformed_X).evaluate(self.grid_points)
        
        if self.method == 'BoxKDE':
            self._knr = kde
            
            if compute_grid:
                self._X_grid, self._grid_density = self._knr.evaluate(self.grid_points)
                self._X_grid = self._inverse_transform_to_principal_component_space(self._X_grid)
        
        else:
            # --------- ROTATE THE GRID BACK ---------
            # After rotation, the grid will not be alix-aligned
            if self._num_dims == 1:
                grid = grid.copy()[:,None]
            
            grid_rot = grid / np.sqrt(self._num_obs) @ np.linalg.inv(self._V @ np.diag(1/self._s))
            
            # --------- RESAMPLE THE GRID ---------
            # We pretend the data is the rotated grid, and the f(x, y) values are weights
            # This is a re-sampling of the KDE onto an axis-aligned grid, and is needed
            # since matplotlib requires axis-aligned grid for plotting.
            # (The problem of plotting on an arbitrary grid is similar to density estimation)
            
            kde = _kde_class[self.method](kernel=self.kernel, norm=self.norm, bw=self._bw)
            self._X_grid, self._grid_density = kde.fit(grid_rot, weights=points).evaluate(self.grid_points)
            
            if self._num_dims == 1:
                self._X_grid = self._X_grid[:,None]
            
            if len(self.bounded_features) > 0:
                self._grid_density *= 2**len(self.bounded_features)    
                self._grid_density[np.any(self._X_grid[:, self.bounded_features] < self._low_bounds,axis=1)] = 0
            
            self._knr = KNeighborsRegressor(n_neighbors=2**len(self.bounded_features), weights='distance')
            self._knr.fit(self._X_grid, self._grid_density)
        
    def plot_bw_opt(self):
        df = pd.DataFrame(self._opt_bw, columns=['bw'])
        df['J'] = self._opt_J
        df.sort_values(by='bw', inplace=True)
        
        plt.plot(df.bw, df.J)
        plt.scatter(df.bw, df.J, label='opt algo')
        plt.vlines(self._bw, ymin=df.J.min(), ymax=df.J.max(), color='red', label='selected value')
        plt.xlabel('h')
        plt.ylabel('J')
        plt.legend()
        
        return(plt)
    
    def predict(self, X):
        if len(X.shape) == 1:
            X = X[:,None]
        
        if self.method == 'BoxKDE':
            X = self._transform_to_principal_component_space(X)
            p = self._knr.predict(X) * 2 ** len(self.bounded_features)
            p[np.any(self._X_grid[:, self.bounded_features] < self._low_bounds,axis=1)] = 0
            return(p)
        
        return(self._knr.predict(X))
    
    def _mirror(self, X):
        # first, a full symmetry is made
        self._low_bounds = X[:, self.bounded_features].min(axis=0)
        
        for idx, feature in enumerate(self.bounded_features):
            X_mirrored = X.copy()
            X_mirrored[:, feature] = 2 * self._low_bounds[idx] - X[:, feature]

            X = np.vstack((X, X_mirrored))
        
        if self.verbose > 0 and len(self.bounded_features) > 0:
            print('mirrored data shape : ', X.shape)
        
        return(X)
        
    def _transform_to_principal_component_space(self, X):
        return(X @ self._V @ np.diag(1 / self._s) * np.sqrt(self._num_obs))
    
    def _inverse_transform_to_principal_component_space(self, X):
        return(X / np.sqrt(self._num_obs) @ np.diag(self._s)  @ self._V.T )
    
    def _compute_bw_ucv(self, X):
        if self.verbose > 0:
            print('\n ==== \n BandWidth computing through UCV method \n ==== \n\n')
        
        bw = np.mean([silvermans_rule(X[:, [k]]) for k in range(self._num_dims)])
        
        self._opt_bw = []
        self._opt_J = []
        bw = optimize.fmin(self._compute_J,
                           bw,
                           (X,),
                           xtol=self.bw_tol,
                           ftol=self.J_tol)
        
        return(float(bw[0]))
    
    
    def _compute_J(self, bw, X):
        if type(bw) is list or type(bw) is np.ndarray:
            bw = bw[0]
        
        if bw < 0:
            return(np.nan)
        
        n = X.shape[0]
        d = X.shape[1]
        
        kde = _kde_class[self.method](kernel=self.kernel, norm=self.norm, bw=bw)
        kde.fit(X)
        
        if self.method=='BoxKDE' and 1==0:
            c = list(combinations(np.arange(n), 2))
            integral_square = 0
            
            V = volume_unit_ball(d=d, p=self.norm)
        
            for i, j in tqdm(c):
                normed_x = p_norm(X[[i,j],:], self.norm)
                d = np.abs(normed_x[0]- normed_x[1])
                if d <= bw / 2:
                    a = bw - d
                    integral_square += a
        
            integral_square = 1 / (n**2 * (bw/2)**(2*d) * V**2) * (n * bw + 2 * integral_square)
            
            knr = kde
        else:
            grid, points = kde.evaluate(self.grid_points)
            integral_square = np.sum(points**2) * np.product(grid.max(axis=0)-grid.min(axis=0)) / points.size
            
            knr = KNeighborsRegressor(n_neighbors=2**len(self.bounded_features),
                                      weights='distance')
            
            if len(grid.shape) == 1:
                grid = grid[:,None]
            
            knr.fit(grid, points)
            
        K0 = _kernel_functions[self.kernel](np.zeros((1, self._num_dims)), bw=bw, norm=self.norm)[0]
        s = n / (n-1) * (knr.predict(X).sum() - K0)
        
        J = integral_square - 2 / n * s
        
        if self.verbose > 0:
            print('bw=', bw, 'J=', J)
        
        self._opt_bw.append(bw)
        self._opt_J.append(J)
        
        return(J)