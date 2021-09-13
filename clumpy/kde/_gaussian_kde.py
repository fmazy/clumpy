#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 20:16:24 2021

@author: frem
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import erf
from multiprocessing import Pool
from sklearn.base import BaseEstimator
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from time import time
from . import bandwidth_selection
from tqdm import tqdm
from ._whitening_transformer import _WhiteningTransformer

class GKDE(BaseEstimator):
    def __init__(self,
                 h=1.0,
                 low_bounded_features=[],
                 high_bounded_features=[],
                 low_bounds = [],
                 high_bounds = [],
                 algorithm='kd_tree',
                 leaf_size=30,
                 support_factor=3,
                 forbid_null_value = False,
                 n_predict_max = 2*10**4,
                 n_jobs=1,
                 preprocessing='whitening',
                 verbose=0):
        self.h = h
        self.low_bounded_features = low_bounded_features
        self.high_bounded_features = high_bounded_features
        self.low_bounds = low_bounds
        self.high_bounds = high_bounds
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.support_factor = support_factor
        self.forbid_null_value = forbid_null_value
        self.n_predict_max = n_predict_max
        self.n_jobs = n_jobs
        self.preprocessing = preprocessing
        self.verbose = verbose
        
        
    def __repr__(self):
        return('GKDE(h='+str(self._h)+')')
    
    def fit(self, X, y=None):
        
        # preprocessing
        if self.preprocessing == 'standard':
            self._preprocessor = StandardScaler()
            self._data = self._preprocessor.fit_transform(X)
        
        elif self.preprocessing == 'whitening':
            self._preprocessor = _WhiteningTransformer()
            self._data = self._preprocessor.fit_transform(X)
        
        else:
            self._data = X
        
        # get data dimensions
        self._n = self._data.shape[0]
        self._d = self._data.shape[1]
        
        # BOUNDARIES INFORMATIONS
        # low bounds
        if self.low_bounds is None or len(self.low_bounds) != len(self.low_bounded_features):
            raise(ValueError("unexpected low bounds value"))
        
        self._low_bounds_b = []
        self._low_bounds_w = []
        
        for id_k, k in enumerate(self.low_bounded_features):
            # set a point within the hyperplane.
            # it is simple : it is defined with one axis vector
            # A = np.zeros((1,self._d))
            # A[0, id_k] = self.low_bounds[id_k]
            
            # w = np.zeros((1,2))
            # w[0,id_k] = 1
            
            A = np.diag(np.ones(2))
            A[:,id_k] = self.low_bounds[id_k]
            
            A_wt = self._preprocessor.transform(A)
            
            w_wt = np.linalg.solve(A_wt.T, np.ones(2))
                        
            b = - np.dot(w_wt, A_wt[0])
            print('A_wt', A_wt, 'w_wt', w_wt)
            print('b', b)
            
            self._low_bounds_b.append(b)
            self._low_bounds_w.append(w_wt)
        
        # high bounds
        if self.high_bounds is None or len(self.high_bounds) != len(self.high_bounded_features):
            raise(ValueError("unexpected low bounds value"))
        
        self._high_bounds_b = []
        self._high_bounds_w = []
        for id_k, k in enumerate(self.high_bounded_features):
            # set a point within the hyperplane.
            # it is simple : it is defined with one axis vector
            # A = np.zeros((1,self._d))
            # A[0, id_k] = self.high_bounds[id_k]
            
            # w = np.zeros((1,2))
            # w[0,id_k] = 1
            A = np.diag(np.ones(2))
            A[:,id_k] = self.high_bounds[id_k]
            
            # w_wt = self._preprocessor.transform(w)[0]
            A_wt = self._preprocessor.transform(A)
            
            w_wt = np.linalg.solve(A_wt.T, np.ones(2))
            
            b = - np.dot(w_wt, A_wt)
            
            self._high_bounds_b.append(b)
            self._high_bounds_w.append(w_wt)
        
        # BANDWIDTH SELECTION
        if type(self.h) is int or type(self.h) is float:
            self._h = float(self.h)
        
        elif type(self.h) is str:
            if self.h == 'scott' or self.h == 'silverman':
                self._h = bandwidth_selection.scotts_rule(X)
            else:
                raise(ValueError("Unexpected bandwidth selection method."))
        else:
            raise(TypeError("Unexpected bandwidth type."))
        
        # NORMALIZATION FACTOR
        self._normalization = 1 / ((2*np.pi)**(self._d/2) * self._h**self._d)
        
        # NEAREST NEIGHBOR FITTING
        self._nn = NearestNeighbors(radius = self._h * self.support_factor,
                                   algorithm = self.algorithm,
                                   leaf_size = self.leaf_size,
                                   n_jobs = self.n_jobs)
        self._nn.fit(self._data)
        
        return(self)
        
    def predict(self, X):
        pdf = self._predict_with_bandwidth(X, self._h)
        return(pdf)
    
    def _predict_with_bandwidth(self, X, h, scaling=True):
        
        id_out_of_low_bounds = np.any(X[:, self.low_bounded_features] < self.low_bounds)
        id_out_of_high_bounds = np.any(X[:, self.high_bounded_features] > self.high_bounds)
        
        if self.preprocessing != 'none':
            X = self._preprocessor.transform(X)
        
        f = np.zeros(X.shape[0])
        
        # STEPS INITIALIZATION
        # requested elements are not estimated alltogether in order to
        # improve numerical computations.
        # the limit is given by n_predict_max
        steps = np.arange(0, X.shape[0], self.n_predict_max)
        if self.verbose > 0:
            steps = tqdm(steps)
        
        # for each set of n_predict_max elements
        for i in steps:
            distances, neighbors_id = self._nn.radius_neighbors(X[i:i+self.n_predict_max], radius=h * self.support_factor, return_distance=True)
            
            if self.n_jobs == 1:
                f[i:i+self.n_predict_max] = np.array([_gaussian(dist/h) for dist in distances])
                
            else:
                pool = Pool(self.n_jobs)
                f[i:i+self.n_predict_max] = np.array(pool.starmap(_gaussian, [(dist/h,) for dist in distances]))
        
        # Normalization
        f *= self._normalization / self._n
        
        # boundary bias correction
        for id_k, k in enumerate(self.low_bounded_features):
            dist = np.abs(np.dot(X, self._low_bounds_w[id_k]) + self._low_bounds_b[id_k]) / np.linalg.norm(self._low_bounds_w[id_k])
            
            omega = np.array([1.5, -1])
            plt.plot([omega[0], (omega+self._low_bounds_w[id_k])[0]], [omega[1], (omega+self._low_bounds_w[id_k])[1]], c='red')
            plt.scatter(X[:,0], X[:,1], s=2, c=dist)
            plt.show()
            
            f /= 1 / 2 * (1 + erf(dist / h / np.sqrt(2)))
            
        # outside bounds : equal to 0
        f[id_out_of_low_bounds] = 0
        f[id_out_of_high_bounds] = 0
            
        # if self.preprocessing == 'standard':
        #     for id_k, k in enumerate(self.low_bounded_features):
        #         id_inside_bounds = X[:,k] >= self._low_bounds[id_k]
        #         f[id_inside_bounds] /= 1 / 2 * (1 + erf((X[id_inside_bounds,k] - self._low_bounds[id_k]) / h / np.sqrt(2)))
                        
        #     for id_k, k in enumerate(self.high_bounded_features):
        #         id_inside_bounds = X[:,k] <= self._high_bounds[id_k]
        #         f[id_inside_bounds] /= 1 / 2 * (1 + erf((self._high_bounds[id_k] - X[id_inside_bounds,k]) / h / np.sqrt(2)))
                
        # elif self.preprocessing == 'whitening':
        #     for id_k, k in enumerate(self.low_bounded_features):
                
        #         a = np.zeros(self._d)
        #         # a[k]
        #         b = np.zeros(self._d)
        #         print(self._low_bounds)
        #         id_inside_bounds = X[:,k] >= self._low_bounds[id_k]
        #         f[id_inside_bounds] /= 1 / 2 * (1 + erf((X[id_inside_bounds,k] - self._low_bounds[id_k]) / h / np.sqrt(2)))
                        
        #     for id_k, k in enumerate(self.high_bounded_features):
        #         id_inside_bounds = X[:,k] <= self._high_bounds[id_k]
        #         f[id_inside_bounds] /= 1 / 2 * (1 + erf((self._high_bounds[id_k] - X[id_inside_bounds,k]) / h / np.sqrt(2)))
        
        # boundary cutting if necessary
        # f[np.any(X[:, self.low_bounded_features] < self._low_bounds, axis=1)] = 0
        # f[np.any(X[:, self.high_bounded_features] > self._high_bounds, axis=1)] = 0
        
        # Preprocessing correction
        if self.preprocessing != 'none':
            f /= np.product(self._preprocessor.scale_)
        
        # if null value is forbiden
        if self.forbid_null_value:
            idx = f == 0.0
            
            new_n = self._n + idx.sum()
            
            f = f * self._n / new_n
            
            min_value = 1 / new_n / self._normalization * _gaussian(0)
            f[f == 0.0] = min_value
        
        return(f)
        
    def J(self):
        
        # on mutualise le calcul de distances.
        # on regarde donc d'abord dans un rayon de 3h puis on prend ce que l'on
        # veut ensuite.
        distances, _ = self._nn.radius_neighbors(self._data, radius=self._h * self.support_factor, return_distance=True)
        
        J = self._normalization * (np.sum([1 / self._n**2 / 2**(self._d/2) * np.sum(np.exp(-1 / 2 * self._h**2 * dist[dist<=self._h * self.support_factor / np.sqrt(2)]**2)) - 2 / (self._n -1) / self._n * np.sum(np.exp(-0.5 * dist**2 / self._h**2)) for dist in distances]) + 2 / (self._n-1) * np.exp(0))
        
        return(J)
    
    def marginal(self,
                 x,
                 k):
        
        X = np.zeros((x.size, self._d))
        X[:,k] = x
        
        if self.preprocessing != 'none':
            X = self._preprocessor.transform(X)
        x = X[:, k]
        
        nn = NearestNeighbors(radius=self._h * self.support_factor,
                              algorithm = self.algorithm,
                              leaf_size=self.leaf_size,
                              n_jobs = self.n_jobs)
        nn.fit(self._data[:,[k]])
        
        distances, _ = nn.radius_neighbors(x[:,None],
                                           radius= self._h * self.support_factor,
                                           return_distance=True)
        
        f = 1 / (2 * np.pi)**(1/2) / self._h / self._n * np.array([_gaussian(dist / self._h) for dist in distances])
        
        if k in self.low_bounded_features:
            id_k = self.low_bounded_features.index(k)
            f /= 1 / 2 * (1 + erf((x - self._low_bounds[id_k]) / self._h / np.sqrt(2)))
            
            f[x < self._low_bounds[id_k]] = 0
        
        if k in self.high_bounded_features:
            id_k = self.high_bounded_features.index(k)
            f /= 1 / 2 * (1 + erf((self._high_bounds[id_k] - x) / self._h / np.sqrt(2)))
            
            f[x > self._high_bounds[id_k]] = 0
        
        if self.preprocessing != 'none':
            f /= np.product(self._preprocessor.scale_[k])
        
        return(f)
    
    def cut_plot(self,
                 x=None,
                 q=None,
                 n_eval=300,
                 linestyle=None,
                 color=None,
                 label_prefix=''):
        # check
        if (x is None and q is None) or (x is not None and q is not None):
            raise(ValueError("Either x or q are expected."))
        
        if x is not None:
            if np.sum(np.array(x) == None) != 1:
                raise(ValueError("Only one undefined features is expected. Example: x=[0.2, None, 0.1, 0.6]"))
            
            label='x='+str(x)
            
            k = x.index(None)
            k_bar = np.delete(np.arange(len(x)), k)
            
            x = self._preprocessor.transform(np.array([x]))[0]
            x[k] = None
            
            
            
        if q is not None:
            if np.sum(np.array(q) == None) != 1:
                raise(ValueError("Only one undefined features is expected. Example: x=[0.2, None, 0.1, 0.6]"))
            
            label='q='+str(q)
            
            k = q.index(None)
            k_bar = np.delete(np.arange(len(q)), k)
            
            q[k] = 0
            
            x = [np.quantile(self._data[:,i], qi) for i, qi in enumerate(q)]
            x = np.array(x)
            x[k] = None
            
            
        
        # print(k_bar)
        X_eval = np.ones((n_eval, self._d))
        X_eval[:, k_bar] *= x[k_bar]
        
        if k in self.low_bounded_features:
            a = self._low_bounds[self.low_bounded_features.index(k)]
        else:
            a = self._data[:,k].min() - self.support_factor * self._h
        
        if k in self.high_bounded_features:
            b = self._high_bounds[self.high_bounded_features.index(k)]
        else:
            b = self._data[:,k].max() + self.support_factor * self._h
        
        X_eval[:, k] = np.linspace(a, b, n_eval)
       
        X_eval = self._preprocessor.inverse_transform(X_eval)
       
        pred = self.predict(X_eval)
        
        plt.plot(X_eval[:,k], pred,
                 label=label_prefix+label,
                 color=color,
                 linestyle=linestyle)
        
        return(X_eval, pred, plt)
    
def _gaussian(x):
    return(np.sum(np.exp(-0.5 * x**2)))

def hyper(X):
   k=np.ones((X.shape[0],1))
   a=np.dot(np.linalg.inv(X), k)
   return(a.T[0])