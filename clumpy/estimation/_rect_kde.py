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
from time import time
import pandas as pd
from matplotlib import pyplot as plt
import noisyopt

class RectKDE():
    def __init__(self,
                 h,
                 p=2,
                 bounded_features=[],
                 h_min = 0.1,
                 h_max = 1,
                 h_step = 0.01,
                 grid_shape = 2**8,
                 integral_tol = 1e-2,
                 algorithm='auto',
                 leaf_size=30,
                 n_jobs=None,
                 verbose=0):
        self.h = h
        self.p = p
        self.bounded_features = bounded_features
        self.h_min = h_min
        self.h_max = h_max
        self.h_step = h_step
        self.grid_shape = grid_shape
        self.integral_tol = 1e-2
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.verbose = verbose
    
    def fit(self, X):
        
        if self.verbose > 0:
            print('input data shape : ', X.shape)
            
            if len(self.bounded_features) > 0:
                print('\nsymetry for bounds')
                print('bounded_features : ', self.bounded_features)
        
        plt.scatter(X[:,0], X[:,1],s=2)
        X = self._mirror(X)
        plt.scatter(X[:,0], X[:,1],s=2)
        
        self._support = [X.min(axis=0) - X.std(axis=0),
                         X.max(axis=0) + X.std(axis=0)]
        
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
        
        self._v = volume_unit_ball(self._d, self.p) * self._whitening_transformer._inverse_transform_det
        
        if type(self.h) is float:
            initial_radius = self.h
        else:
            initial_radius = (self.h_max + self.h_min) / 2
        
        self._nn = NearestNeighbors(radius = initial_radius,
                                    algorithm = self.algorithm,
                                    leaf_size = self.leaf_size,
                                    p = self.p,
                                    n_jobs = self.n_jobs)
        self._nn.fit(self._data)
        
        if type(self.h) is str:
            if self.h == 'UCV':
                self._h = self._compute_h_through_ucv()
            elif self.h == 'UCV_mc':
                self._h = self._compute_h_through_ucv(montecarlo=True)
            else:
                raise(TypeError("Unexpected h parameter type. Should be a float or {'UCV', 'UCV_mc'}."))
        elif type(self.h) is float:
            self._h = self.h
        else:
            raise(TypeError("Unexpected h parameter type. Should be a float or {'UCV', 'UCV_mc'}."))
        
        if self.verbose > 0:
            print('\nNearest neighbors training...')
        
        self._nn.radius = self._h
        
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
    
    def predict(self, X):
        X = self._whitening_transformer.transform(X)
        
        neigh_ind = self._nn.radius_neighbors(X, return_distance=False)
        
        density = np.array([ni.size for ni in neigh_ind])
        
        density = density / ( self._n * self._v * self._h**self._d)
        
        density[np.any(X[:, self.bounded_features] < self._low_bounds, axis=1)] = 0
        
        density = density * 2 ** len(self.bounded_features)
        
        return(density)
    
    def grid_predict(self):
        X_grid = self._create_grid()
        
        density = self.predict(X_grid)
        
        integral = density.sum() * np.product(X_grid.max(axis=0)-X_grid.min(axis=0)) / X_grid.shape[0]
        
        if self.verbose > 0:
            print('integral=',integral)
        
        _check_integral_close_to_1(integral)
        
        return(X_grid, density)
        
    def _create_grid(self):
        if type(self.grid_shape) is int:
            grid_shape = (np.ones(self._d) * self.grid_shape).astype(int)
        else:
            grid_shape = self.grid_shape
        
        xk = np.meshgrid(*(np.linspace(self._support[0][k],self._support[1][k], grid_shape[k]) for k in range(self._d)))
        X_grid = np.vstack([xki.flat for xki in xk]).T
        
        return(X_grid)
# new_X_grid  = np.vstack([x0.flat, x1.flat]).T
    
    # def grid_predict(self, grid_points, return_integral=False):
    #     support_min, support_max = self._support_in_transformed_space()
        
    #     xk = np.meshgrid(*(np.linspace(support_min[k], support_max[k], grid_points[k]) for k in range(self._d)))
    #     X_grid = np.vstack([xk[k].flat for k in range(self._d)]).T
        
    #     X_grid = self._whitening_transformer.inverse_transform(X_grid)
        
    #     pred_grid = self.predict(X_grid)
        
    #     integral = pred_grid.sum() * np.product(support_max - support_min) / pred_grid.size
        
    #     print((pred_grid**2).sum() * np.product(support_max - support_min) / pred_grid.size)
        
    #     # return only within the bounds
    #     # idx = np.all(X_grid[:, self.bounded_features] >= self._low_bounds, axis=1)
    #     # X_grid = X_grid[idx]
    #     # pred_grid = pred_grid[idx]
        
    #     if return_integral:
    #         return(X_grid, pred_grid, integral)
    #     #     if return_integral_out_of_bounds:
    #     #         idx_out_of_bounds = np.any(X_grid[:,self.bounded_features] < self._low_bounds, axis=1)
    #     #         integral_out_of_bounds = pred_grid[idx_out_of_bounds].sum() * np.product(support_max - support_min) / pred_grid.size
                
    #     #         return(X_grid, pred_grid, integral, integral_out_of_bounds)
    #     #     return(X_grid, pred_grid, integral)
    #     return(X_grid, pred_grid)
    
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
    
    # def _cut_mirror(self, X):
        
    #     X_inv_trans = self._whitening_transformer.inverse_transform(X)
        
    #     X = X[np.all(X_inv_trans[:, self.bounded_features] >= self._low_bounds - self._hmax, axis=1)]
        
    #     if self.verbose>0:
    #         print('cutted mirror data shape : ', X.shape)
        
    #     return(X)
    
    def _compute_h_through_ucv(self, montecarlo=False):
        if montecarlo:
            X_grid = self._create_grid()
        else:
            X_grid = None
        
        self._opt_h = np.arange(start = self.h_min,
                           stop = self.h_max,
                           step = self.h_step)
        self._opt_J = []
        
        st = time()
        for h in self._opt_h:
            self._opt_J.append(self._compute_J(h, X_grid, real_scale=False))
            
            if self.verbose>0:
                print(h, self._opt_J[-1])
        
        self._opt_time = time() - st
        self._opt_J = np.array(self._opt_J)
        
        return(self._opt_h[np.argmin(self._opt_J)])
        
    def _compute_J(self, h, X_grid=None, real_scale=True):
        if type(h) is np.ndarray:
            h = float(h)
        
        integral_squared = self._compute_integral_squared(h, X_grid, real_scale=real_scale)
        
        leave_one_out_esperance = self._compute_leave_one_out_esperance(h, real_scale=real_scale)
        
        J = integral_squared - 2 * leave_one_out_esperance
        
        return(J)
    
    def _compute_integral_squared(self, h, X_grid=None, real_scale=True):
        if X_grid is None:
            return(self._compute_exact_integral_squared(h, real_scale=real_scale))
        else:
            return(self._compute_mc_integral_squared(h, X_grid, real_scale=real_scale))
    
    def _compute_mc_integral_squared(self, h, X_grid, real_scale=True):
        
        mc_coef = np.product(X_grid.max(axis=0)-X_grid.min(axis=0)) / X_grid.shape[0]
                
        X_grid = self._whitening_transformer.transform(X_grid)
             
        neigh_ind = self._nn.radius_neighbors(X=X_grid,
                                              radius=h,
                                              return_distance=False)
        
        p_grid = np.array([ni.size for ni in neigh_ind])
        p_grid = p_grid / ( self._n * self._v * h**self._d)
        
        integral = p_grid.sum() * mc_coef
        
        _check_integral_close_to_1(integral, eps=self.integral_tol)
        
        integral_squared = (p_grid**2).sum() * mc_coef
        
        if not real_scale:
            integral_squared *= self._n * self._v
        
        return(integral_squared)
    
    def _compute_exact_integral_squared(self, h, real_scale=True):
        neigh_dist, neigh_ind = self._nn.radius_neighbors(radius= 2 * h,
                                                          return_distance=True)
        
        hypersphere_intersection_volume = 2*np.sum([Vn(h, d/2, self._d).sum() for d in neigh_dist]) * self._whitening_transformer._inverse_transform_det / 2
        
        integral_squared = 1 / (h**self._d)
        integral_squared += 2 / (self._n * h**(2 * self._d) * self._v) * hypersphere_intersection_volume
        
        if real_scale:
            integral_squared /= self._n * h**(self._d) * self._v
        
        return(integral_squared)
        
    def _compute_leave_one_out_esperance(self, h, real_scale=True):
        neigh_ind = self._nn.radius_neighbors(radius = h,
                                        return_distance=False)
        
        s = np.array([ni.size for ni in neigh_ind]).sum() / ((self._n - 1) * h**self._d)
        
        if real_scale:
            s /= self._n * self._v
        
        return(s)

    def plot_h_opt(self):
        df = pd.DataFrame(self._opt_h, columns=['h'])
        df['J'] = self._opt_J
        df.sort_values(by='h', inplace=True)
        
        plt.plot(df.h, df.J, label='opt algo')
        # plt.scatter(df.h, df.J, label='opt algo', s=5)
        plt.vlines(self._h, ymin=df.J.min(), ymax=df.J.max(), color='red', label='selected value')
        plt.xlabel('h')
        plt.ylabel('$\hat{J}$')
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
        
        self._transform_det = np.abs(np.linalg.det(self._transform_matrix))
        self._inverse_transform_det = np.abs(np.linalg.det(self._inverse_transform_matrix))
        
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

def _check_integral_close_to_1(integral, eps=1e-2):
    if np.abs(1-integral) > eps:
        print("/!\ WARNING /!\ Integral="+str(integral)+"\nwhich is too far from 1.\nThe grid density should be increased.")
        return(False)
    else:
        return(True)