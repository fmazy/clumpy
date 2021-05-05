#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 09:24:01 2021

@author: frem
"""

import numpy as np
from sklearn.neighbors import KDTree, BallTree, NearestNeighbors
from scipy.special import gamma, betainc
from scipy.optimize import fmin
from itertools import combinations
from tqdm import tqdm
from time import time
import pandas as pd
from matplotlib import pyplot as plt
import noisyopt

_algorithm_class = {
    'kd_tree':KDTree,
    'ball_tree':BallTree
    }

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
                 algorithm='kd_tree',
                 leaf_size=30,
                 dualtree=False,
                 verbose=0):
        self.h = h
        self.p = p
        self.bounded_features = bounded_features
        self.h_min = h_min
        self.h_max = h_max
        self.h_step = h_step
        self.grid_shape = grid_shape
        self.integral_tol = integral_tol
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.dualtree = dualtree
        self.verbose = verbose
    
    def fit(self, X):
        
        if self.verbose > 0:
            print('input data shape : ', X.shape)
            
            if len(self.bounded_features) > 0:
                print('\nsymetry for bounds')
                print('bounded_features : ', self.bounded_features)
        
        #============================================================
        # 1. Data operations
        #============================================================
        
        # --------------------------------
        # 1.1. Mirror
        # --------------------------------
        
        # mirror the data is case of bounded features
        # for now, it is a full symetry
        self._first_mirror_id = X.shape[0]
        X, self._low_bounds = _mirror(X, bounded_features=self.bounded_features)
        
        if self.verbose > 0 and len(self.bounded_features) > 0:
            print('mirrored data shape : ', X.shape)
        
        # --------------------------------
        # 1.2. Support
        # --------------------------------
        
        # the support is used for grid creating
        # (notably in 'UCV_mc' h selection method)
        self._support = [X.min(axis=0) - X.std(axis=0),
                         X.max(axis=0) + X.std(axis=0)]
        self._support[0][self.bounded_features] = self._low_bounds - X[:, self.bounded_features].std(axis=0)/2
        
        # --------------------------------
        # 1.3. Whitening transformation
        # --------------------------------
        
        # whitening transformation
        self._whitening_transformer = _WhiteningTransformer()
        # X is ereased
        X = self._whitening_transformer.fit_transform(X)
        
        # n and d are set here
        # n is not equal to self._data.shape[0] is case of mirror
        # due to mirror data selection
        self._n = X.shape[0]
        self._d = X.shape[1]
        
        # volume to apply to KDE according to the whitening transformation
        self._v = volume_unit_ball(self._d, self.p) * self._whitening_transformer._inverse_transform_det
                
        # --------------------------------
        # 1.4. Mirror data selection
        # --------------------------------
        
        if len(self.bounded_features) > 0:
            # get hmax if h is given
            if type(self.h) is float:
                # if no bandwidth selection, the h radius is enought
                radius_ms = self.h
            else:
                # if bandwidth selection, 2 * hmax radius are required
                # for exact integral squared
                radius_ms = self.h_max * 2
            
            X = _mirror_data_selection(X=X,
                                       algorithm=self.algorithm,
                                       leaf_size=self.leaf_size,
                                       first_mirror_id=self._first_mirror_id,
                                       radius=radius_ms)
            
            if self.verbose > 0:
                print('selected mirrored data shape : ', X.shape)
        
        # --------------------------------
        # 1.5. Data conclusion
        # --------------------------------
        # all data operations are made
        self._data = X
        
        #============================================================
        # 2. First Nearest Neighbors setting
        #============================================================
        
        # The KDE nearest neighbors tree is set a first time
        if self.verbose > 0:
            print('\nNearest neighbors tree training...')
        # Nearest Neighbors tree
        if self.algorithm not in _algorithm_class.keys():
            raise(ValueError("Unexpected algorithm parameter '"+str(self.algorithm)+"'. It should belong to {'kd_tree', 'ball_tree'}."))
        self._tree = _algorithm_class[self.algorithm](self._data,
                                                      leaf_size=self.leaf_size)
        if self.verbose > 0:
            print('...done')
        
        #============================================================
        # 3. Bandwidth selection
        #============================================================
        
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
                
        #============================================================
        # 4. Final training
        #============================================================
        # If their are mirrored data and the bandwidth has been selected,
        # training data should be adujsted to the new bandwidth
        # then, radius neighbors are searched for mirrored data
        if len(self.bounded_features) > 0 and type(self.h) is not float:
            self._data = _mirror_data_selection(X=self._data,
                                                algorithm=self.algorithm,
                                                leaf_size=self.leaf_size,
                                                first_mirror_id=self._first_mirror_id,
                                                radius=self._h)
            
            # The KDE nearest neighbors tree is set a first time
            if self.verbose > 0:
                print('\nFinal nearest neighbors tree training...')
            # Nearest Neighbors tree
            self._tree = _algorithm_class[self.algorithm](self._data,
                                                          leaf_size=self.leaf_size)
            if self.verbose > 0:
                print('...done')
        
        return(self)
    
    def predict(self, X, h=None):
        if h is None:
            h = self._h
        
        # get out_of_bounds indices
        out_of_bounds_ind = np.any(X[:, self.bounded_features] < self._low_bounds, axis=1)
        
        # transform X to whitening space
        X = self._whitening_transformer.transform(X)
        
        # count neighbors
        density = self._tree.query_radius(X=X,
                                          r=h,
                                          count_only=True)
        
        # divide for integral closure
        density = density / ( self._n * self._v * h**self._d)
        
        # scale in case of mirrors
        density[out_of_bounds_ind] = 0
        density = density * 2 ** len(self.bounded_features)
        
        return(density)
    
    def grid_predict(self, h=None, grid_shape=None):
        if h is None:
            h = self._h
        
        if grid_shape is None:
            grid_shape = self.grid_shape
        
        # create a grid
        X_grid = self._create_grid(grid_shape = grid_shape)
        
        # get grid density
        density = self.predict(X_grid, h=h)
        
        # compute integral
        integral = density.sum() * np.product(X_grid.max(axis=0)-X_grid.min(axis=0)) / X_grid.shape[0]
        
        if self.verbose > 0:
            print('integral=',integral)
        
        # check integral validity
        _check_integral_close_to_1(integral, eps=self.integral_tol)
        
        return(X_grid, density)
    
    def _create_grid(self, grid_shape=None):
        if grid_shape is None:
            grid_shape = self.grid_shape
        
        # if grid_shape is int, use this value for all dimensions
        if type(grid_shape) is int:
            grid_shape = (np.ones(self._d) * grid_shape).astype(int)
        
        # create linear mesh grid
        xk = np.meshgrid(*(np.linspace(self._support[0][k],self._support[1][k], grid_shape[k]) for k in range(self._d)))
        X_grid = np.vstack([xki.flat for xki in xk]).T
        
        return(X_grid)
    
    def _compute_h_through_ucv(self, montecarlo=False, real_scale=False):
        if montecarlo:
            # if montecarlo, create a grid.
            # the grid is thus created only one time.
            X_grid = self._create_grid()
        else:
            X_grid = None
        
        # linear bandwidth variation 
        self._opt_h = np.arange(start = self.h_min,
                           stop = self.h_max,
                           step = self.h_step)
        self._opt_J = []
        
        # start time for execution time
        st = time()
        for h in self._opt_h:
            # compute J and append it to _opt_J
            # the real scale is set to False. The real value of J is not
            # required. Comparisons are enought
            self._opt_J.append(self._compute_J(h, X_grid, real_scale=real_scale))
            
            if self.verbose>0:
                print(h, self._opt_J[-1])
        
        # execution time
        self._opt_time = time() - st
        
        # opt_J as a numpy array
        self._opt_J = np.array(self._opt_J)
        
        return(self._opt_h[np.argmin(self._opt_J)])
        
    def _compute_J(self, h, X_grid=None, real_scale=True):
        # compute integral squared. If montecarlo, X_grid is not None
        integral_squared = self._compute_integral_squared(h, X_grid, real_scale=real_scale)
        
        # compute leave one out esperance
        leave_one_out_esperance = self._compute_leave_one_out_esperance(h, real_scale=real_scale)
        
        # compute J
        J = integral_squared - 2 * leave_one_out_esperance
        
        return(J)
    
    def _compute_integral_squared(self, h, X_grid=None, real_scale=True):
        # montecarlo switch
        if X_grid is None:
            return(self._compute_exact_integral_squared(h, real_scale=real_scale))
        else:
            return(self._compute_mc_integral_squared(h, X_grid, real_scale=real_scale))
    
    def _compute_mc_integral_squared(self, h, X_grid, real_scale=True):
        # montecarlo coefficient according to the original space
        mc_coef = np.product(X_grid.max(axis=0)-X_grid.min(axis=0)) / X_grid.shape[0]
                
        # count neighbors through the predict function with set h
        p_grid = self.predict(X_grid, h=h)
        
        # compute integral
        integral = p_grid.sum() * mc_coef
        
        # integral check
        # a warning message can be displayed but it is not considered as
        # an error
        _check_integral_close_to_1(integral, eps=self.integral_tol)
        
        # compute integral squared
        integral_squared = (p_grid**2).sum() * mc_coef
        
        # mirrors consideration
        # the p_grid has already been scaled for mirror considerations
        # It have to been then downscaled according to formulas.
        # indeed, the integral is made on the whole support
        # without mirror considerations.
        integral_squared /= 2**len(self.bounded_features)
        
        
        if not real_scale:
            # in case of no real scale, some scaling are required
            integral_squared *= self._n * self._v
        
        return(integral_squared)
    
    def _compute_exact_integral_squared(self, h, real_scale=True):
        # get neighbors distances
        indices, distances = self._tree.query_radius(X = self._data[:self._first_mirror_id],
                                                      r = 2 * h,
                                                      return_distance=True)
        
        # compute hypersphere intersection volume
        # see https://math.stackexchange.com/questions/162250/how-to-compute-the-volume-of-intersection-between-two-hyperspheres
        # all pairs of points are taken 2 times. it is in accordance
        # with the integral squared formula
        hypersphere_intersection_volume = 2*np.sum([Vn(h, dist/2, self._d).sum() for dist in distances]) * self._whitening_transformer._inverse_transform_det
        # scale returned volume for mirrors considerations
        hypersphere_intersection_volume *= 2**len(self.bounded_features)
        
        # integral closure
        integral_squared = 1 / (self._n * h**(2 * self._d) * self._v) * hypersphere_intersection_volume
        
        if real_scale:
            # if real scale, some divisions are required
            integral_squared /= self._n * self._v
        
        return(integral_squared)
        
    def _compute_leave_one_out_esperance(self, h, real_scale=True):
        # count pairs of points
        s = self._tree.two_point_correlation(X=self._data[:self._first_mirror_id],
                                             r=h,
                                             dualtree=self.dualtree)[0]
        # one should remove auto-paired points since the data and the training
        # set are the same
        s -= self._first_mirror_id
        
        # mirror considerations
        s *= 2**len(self.bounded_features)
        
        # integral closure
        s = s / ((self._n - 1) * h**self._d)
        
        if real_scale:
            # if real scale, some divisions are required
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
    
def _mirror(X, bounded_features):
    low_bounds = X[:, bounded_features].min(axis=0)
    
    for idx, feature in enumerate(bounded_features):
        X_mirrored = X.copy()
        X_mirrored[:, feature] = 2 * low_bounds[idx] - X[:, feature]

        X = np.vstack((X, X_mirrored))
    
    return(X, low_bounds)

def _mirror_data_selection(X, algorithm, leaf_size, first_mirror_id, radius):
    # mirror data selection
    
    # tree_ms -> nearest neighbors tree mirror selection
    # trained through original data
    tree_ms = _algorithm_class[algorithm](X[:first_mirror_id],
                                          leaf_size=leaf_size)
    # count neighbors for mirrored data
    count_neighbors = tree_ms.query_radius(X=X[first_mirror_id:],
                                           r=radius,
                                           count_only=True)
    # only pixels with neighbors are kept
    ind_to_keep = first_mirror_id + np.arange(len(count_neighbors))[count_neighbors > 0]
    # selected mirrored data
    X = np.vstack((X[:first_mirror_id],
                    X[ind_to_keep]))
    
    return(X)
        
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

