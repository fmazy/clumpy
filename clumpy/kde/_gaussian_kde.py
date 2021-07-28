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

class GKDE(BaseEstimator):
    def __init__(self,
                 h=1.0,
                 low_bounded_features=[],
                 high_bounded_features=[],
                 low_bounds = None,
                 high_bounds = None,
                 algorithm='kd_tree',
                 leaf_size=30,
                 support_factor=3,
                 standard_scaler = True,
                 forbid_null_value = False,
                 n_jobs=1,
                 verbose=0):
        self.h = h
        self.low_bounded_features = low_bounded_features
        self.high_bounded_features = high_bounded_features
        self.low_bounds = low_bounds
        self.high_bounds = high_bounds
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.support_factor = support_factor
        self.standard_scaler = standard_scaler
        self.forbid_null_value = forbid_null_value
        self.n_jobs = n_jobs
        self.verbose = verbose
        
    def __repr__(self):
        return('GKDE(h='+str(self._h)+')')
    
    def fit(self, X, y=None):
        if self.standard_scaler:
            self._standard_scaler = StandardScaler()
            self._data = self._standard_scaler.fit_transform(X)
        else:
            self._data = X
            
        self._n = self._data.shape[0]
        self._d = self._data.shape[1]
        
        if self.low_bounds is None:
            self._low_bounds = self._data[:, self.low_bounded_features].min(axis=0)
        else:
            if self.standard_scaler:
                lb = np.zeros(self._d)
                lb[self.low_bounded_features] = self.low_bounds
                self._low_bounds = self._standard_scaler.transform(lb[None, :])
            else:
                self._low_bounds = self.low_bounds
        
        if self.high_bounds is None:
            self._high_bounds = self._data[:, self.high_bounded_features].max(axis=0)
        else:
            if self.standard_scaler:
                hb = np.zeros(self._d)
                hb[self.high_bounded_features] = self.high_bounds
                self._high_bounds = self._standard_scaler.transform(hb[None, :])
            else:
                self._high_bounds = self.high_bounds
        
        # bandwidth selection
        if type(self.h) is int or type(self.h) is float:
            self._h = float(self.h)
        
        elif type(self.h) is str:
            if self.h == 'scott':
                self._h = bandwidth_selection.scotts_rule(X)
            else:
                raise(ValueError("Unexpected bandwidth selection method."))
        else:
            raise(TypeError("Unexpected bandwidth type."))
        
        self._normalization = 1 / ((2*np.pi)**(self._d/2) * self._h**self._d)
        
        self._nn = NearestNeighbors(radius = self._h * self.support_factor,
                                   algorithm = self.algorithm,
                                   leaf_size = self.leaf_size,
                                   n_jobs = self.n_jobs)
        self._nn.fit(self._data)
        
        return(self)
        
    def predict(self, X):
        
        if self.standard_scaler:
            st = time()
            X = self._standard_scaler.transform(X)
            if self.verbose > 0:
                print('ss', time()-st)
                
        pdf = self._predict_with_bandwidth(X, self._h)
        
        # if null value is forbiden
        if self.forbid_null_value:
            idx = pdf == 0.0
            
            new_n = self._n + idx.sum()
            
            pdf = pdf * self._n / new_n
            
            min_value = 1 / new_n / self._normalization * _gaussian(0)
            pdf[pdf == 0.0] = min_value
        
        return(pdf)
    
    def _predict_with_bandwidth(self, X, h, scaling=True):
        
        st = time()
        distances, neighbors_id = self._nn.radius_neighbors(X, radius=h * self.support_factor, return_distance=True)
        if self.verbose > 0:
            print('neighbors', time()-st)
        
        
        st = time()
        if self.n_jobs == 1:
            f = np.array([_gaussian(dist/h) for dist in distances])
            
        else:
            pool = Pool(self.n_jobs)
            f = pool.starmap(_gaussian, [(dist/h,) for dist in distances])
            f = np.array(f)
            
        f *= self._normalization / self._n
            
        if self.verbose > 0:
            print('gaussian', time()-st)
        
        st = time()
        # boundary bias correction
        for id_k, k in enumerate(self.low_bounded_features):
            f /= 1 / 2 * (1 + erf((X[:,k] - self._low_bounds[id_k]) / h / np.sqrt(2)))
                    
        for id_k, k in enumerate(self.high_bounded_features):
            f /= 1 / 2 * (1 + erf((self._high_bounds[id_k] - X[:,k]) / h / np.sqrt(2)))
        
        if self.verbose > 0:
            print('correction', time()-st)
        
        st = time()
        # boundary cutting if necessary
        f[np.any(X[:, self.low_bounded_features] < self._low_bounds, axis=1)] = 0
        f[np.any(X[:, self.high_bounded_features] > self._high_bounds, axis=1)] = 0
        if self.verbose > 0:
            print('cutting', time()-st)
        
        st = time()
        if self.standard_scaler:
            f /= np.product(self._standard_scaler.scale_)
        if self.verbose > 0:
            print('ss correction', time()-st)
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
        
        if self.standard_scaler:
            X = self._standard_scaler.transform(X)
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
        
        if self.standard_scaler:
            f /= np.product(self._standard_scaler.scale_[k])
        
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
            
            x = self._standard_scaler.transform(np.array([x]))[0]
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
       
        X_eval = self._standard_scaler.inverse_transform(X_eval)
       
        pred = self.predict(X_eval)
        
        plt.plot(X_eval[:,k], pred,
                 label=label_prefix+label,
                 color=color,
                 linestyle=linestyle)
        
        return(X_eval, pred, plt)
    
def _gaussian(x):
    return(np.sum(np.exp(-0.5 * x**2)))