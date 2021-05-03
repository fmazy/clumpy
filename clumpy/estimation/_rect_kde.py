#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 09:24:01 2021

@author: frem
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import gamma, betainc
from scipy.optimize import fmin
from itertools import combinations
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt

class RectKDE():
    def __init__(self,
                 h,
                 hmax=None,
                 p=2,
                 bounded_features=[],
                 grid_points=2**8,
                 htol=1e-3,
                 Jtol=1e-2,
                 verbose=0):
        self.h = h
        self.hmax=hmax
        self.p = p
        self.bounded_features=bounded_features
        self.grid_points = grid_points
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
        
        # if self.hmax is None and type(self.h) is float:
            # self._hmax = self.h
        
        # else:
            # take the silverman value times 3
            # self._hmax = 3 * (self._n * 3 / 4.0) ** (-1 / 5)
        
        # X = self._cut_mirror(X)
        
        self._data = X
        
        self._n = self._data.shape[0]
        self._d = self._data.shape[1]
        
        if type(self.h) is str:
            if self.h == 'UCV':
                self._h = self._compute_h_through_ucv()
        elif type(self.h) is float:
            self._h = self.h
        else:
            raise(TypeError("Unexpected h parameter type. Should be a float or 'UCV'."))
        
        if self.verbose > 0:
            print('\nNearest neighbors training...')
        
        self._nn = NearestNeighbors(radius=self._h, p=self.p)
        self._nn.fit(self._data)
        
        if self.verbose > 0:
            print('...done')
        
        # mirror coefficient ?
        # if len(self.bounded_features) > 0:
        #     if type(self.grid_points) is int:
        #         grid_points = (np.ones(self._d) * self.grid_points).astype(int)
            
        #     if self.verbose > 0:
        #         print('\nComputing mirror coefficient')
        #         print('Grid points : ', grid_points)
        #         print('Estimating grid density...')
            
        #     self._mirror_coef = None
        #     X_grid, pred_grid, integral, integral_out_of_bounds = self.grid_predict(grid_points,
        #                                                                             return_integral=True,
        #                                                                             return_integral_out_of_bounds=True)
            
        #     self._mirror_coef = 1 / (1 - integral_out_of_bounds)
            
        #     if self.verbose > 0:
        #         print('...done')    
        #         print('Integral : ', integral)
        #         if integral < 0.99:
        #             print('/!\ WARNING /!\ The integral is low. The grid points should be increased.')
        #         print('Integral out of bounds : ', integral_out_of_bounds)
        #         print('Mirror coef : ', self._mirror_coef)
            
        return(self)
    
    def predict(self, X=None):
        if X is not None:
            X = self._whitening_transformer.transform(X)
        
        neigh_ind = self._nn.radius_neighbors(X, return_distance=False)
        volume = volume_unit_ball(d=self._d, p=self.p)
        
        density = np.array([ni.size for ni in neigh_ind])
        density = density / ( self._n * self._h**self._d * volume * self._whitening_transformer._inverse_transform_det )
        
        density[np.any(X[:, self.bounded_features] < self._low_bounds, axis=1)] = 0
        
        density = density * 2 ** len(self.bounded_features)
        
        return(density)
    
    def grid_predict(self, grid_points, return_integral=False):
        support_min, support_max = self._support_in_transformed_space()
        
        xk = np.meshgrid(*(np.linspace(support_min[k], support_max[k], grid_points[k]) for k in range(self._d)))
        X_grid = np.vstack([xk[k].flat for k in range(self._d)]).T
        
        X_grid = self._whitening_transformer.inverse_transform(X_grid)
        
        pred_grid = self.predict(X_grid)
        
        integral = pred_grid.sum() * np.product(support_max - support_min) / pred_grid.size
        
        print((pred_grid**2).sum() * np.product(support_max - support_min) / pred_grid.size)
        
        # return only within the bounds
        # idx = np.all(X_grid[:, self.bounded_features] >= self._low_bounds, axis=1)
        # X_grid = X_grid[idx]
        # pred_grid = pred_grid[idx]
        
        if return_integral:
            return(X_grid, pred_grid, integral)
        #     if return_integral_out_of_bounds:
        #         idx_out_of_bounds = np.any(X_grid[:,self.bounded_features] < self._low_bounds, axis=1)
        #         integral_out_of_bounds = pred_grid[idx_out_of_bounds].sum() * np.product(support_max - support_min) / pred_grid.size
                
        #         return(X_grid, pred_grid, integral, integral_out_of_bounds)
        #     return(X_grid, pred_grid, integral)
        return(X_grid, pred_grid)
    
    def _mirror(self, X):
        self._low_bounds = X[:, self.bounded_features].min(axis=0)
        
        for idx, feature in enumerate(self.bounded_features):
            X_mirrored = X.copy()
            X_mirrored[:, feature] = 2 * self._low_bounds[idx] - X[:, feature]

            X = np.vstack((X, X_mirrored))
        
        return(X)
        
        if self.verbose > 0 and len(self.bounded_features) > 0:
            print('mirrored data shape : ', X.shape)
            
        return(X)
    
    def _cut_mirror(self, X):
        
        X_inv_trans = self._whitening_transformer.inverse_transform(X)
        
        X = X[np.all(X_inv_trans[:, self.bounded_features] >= self._low_bounds - self._hmax, axis=1)]
        
        if self.verbose>0:
            print('cutted mirror data shape : ', X.shape)
        
        return(X)
    
    def _support_in_transformed_space(self):
        support_min = self._data.min(axis=0) - self._h
        support_max = self._data.max(axis=0) + self._h
        
        return(support_min, support_max)
    
    def _compute_h_through_ucv(self):
        # silverman rule as h start with sigma=1 (due to the isotrope transformation)
        sigma = 1
        h_start = sigma * (self._n * 3 / 4.0) ** (-1 / 5) * 2
        
        self._opt_h = []
        self._opt_J = []
        h = fmin(self._compute_J,
                    h_start,
                    xtol=self.htol,
                    ftol=self.Jtol)
        
        return(float(h[0]))
        
    def _compute_J(self, h):
        if type(h) is np.ndarray:
            h = float(h)
        
        integral_squared = self._compute_integral_squared(h)
        
        leave_one_out_esperance = self._compute_leave_one_out_esperance(h)
        
        J = integral_squared - 2 * leave_one_out_esperance
        
        print(h, J)
        
        # self._opt_h.append(h)
        # self._opt_J.append(J)
        
        return(leave_one_out_esperance)
    
    def _compute_integral_squared(self, h):
        volume = volume_unit_ball(d=self._d, p=self.p)
        
        neigh_dist, neigh_ind = self._nn.radius_neighbors(radius=2 * h, return_distance=True)
        
        hypersphere_intersection_volume = 2*np.sum([Vn(h, d/2, self._d).sum() for d in neigh_dist]) * self._whitening_transformer._inverse_transform_det
        
        integral_squared = 1 / (self._n * h**(self._d) * volume * self._whitening_transformer._inverse_transform_det) + 2 / (self._n**2 * h**(2*self._d) * volume**2 * self._whitening_transformer._inverse_transform_det**2) * hypersphere_intersection_volume / 2
        
        return(integral_squared)
    
    def _compute_leave_one_out_esperance(self, h):
        volume = volume_unit_ball(d=self._d, p=self.p)
        
        neigh_ind = self._nn.radius_neighbors(radius=h, return_distance=False)
        
        s = np.array([ni.size for ni in neigh_ind]) / ((self._n - 1) * self._h**self._d * volume * self._whitening_transformer._inverse_transform_det)
        
        return(np.mean(s))

    def plot_h_opt(self):
        df = pd.DataFrame(self._opt_h, columns=['h'])
        df['J'] = self._opt_J
        df.sort_values(by='h', inplace=True)
        
        plt.plot(df.h, df.J)
        plt.scatter(df.h, df.J, label='opt algo')
        plt.vlines(self._h, ymin=df.J.min(), ymax=df.J.max(), color='red', label='selected value')
        plt.xlabel('h')
        plt.ylabel('J')
        plt.legend()
        
        return(plt)
        
class _WhiteningTransformer():
    def fit(self, X):
        self._mean = X.mean(axis=0)
        
        self._num_obs = X.shape[0]
        
        _, self._s, Vt = np.linalg.svd(X - self._mean, full_matrices=False)
        self._V = Vt.T
        
        self._transform_matrix = self._V @ np.diag(1 / self._s) * np.sqrt(self._num_obs-1)
        self._inverse_transform_matrix = np.diag(self._s)  @ self._V.T / np.sqrt(self._num_obs-1)
        
        self._transform_det = np.linalg.det(self._transform_matrix)
        self._inverse_transform_det = np.linalg.det(self._inverse_transform_matrix)
        
        return(self)
        
    def transform(self, X):
        X = X - self._mean
        return(X @ self._transform_matrix)
    
    def inverse_transform(self, X):
        X = X @ self._inverse_transform_matrix
        return(X + self._mean)
    
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

def p_norm(x, p):
    """
    The p-norm of an array of shape (obs, dims)

    Examples
    --------
    >>> x = np.arange(9).reshape((3, 3))
    >>> p = 2
    >>> np.allclose(p_norm(x, p), euclidean_norm(x))
    True
    """
    if np.isinf(p):
        return infinity_norm(x)
    elif p == 2:
        return euclidean_norm(x)
    elif p == 1:
        return taxicab_norm(x)
    return np.power(np.power(np.abs(x), p).sum(axis=1), 1 / p)

def euclidean_norm(x):
    """
    The 2 norm of an array of shape (obs, dims)
    """
    return np.sqrt((x * x).sum(axis=1))


def euclidean_norm_sq(x):
    """
    The squared 2 norm of an array of shape (obs, dims)
    """
    return (x * x).sum(axis=1)


def infinity_norm(x):
    """
    The infinity norm of an array of shape (obs, dims)
    """
    return np.abs(x).max(axis=1)


def taxicab_norm(x):
    """
    The taxicab norm of an array of shape (obs, dims)
    """
    return np.abs(x).sum(axis=1)

def Vn(r, a, n):
    return(1/2*np.pi**(n/2)*r**2*betainc((n+1)/2, 1/2, 1-a**2/r**2)/gamma(n/2+1))