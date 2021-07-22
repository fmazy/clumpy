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

class GKDE(BaseEstimator):
    def __init__(self,
                 h=1.0,
                 low_bounded_features=[],
                 high_bounded_features=[],
                 algorithm='kd_tree',
                 leaf_size=30,
                 support_factor=3,
                 standard_scaler=True,
                 n_jobs=1):
        self.h = h
        self.low_bounded_features = low_bounded_features
        self.high_bounded_features = high_bounded_features
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.support_factor = support_factor
        self.standard_scaler = standard_scaler
        self.n_jobs=n_jobs
        
    def __repr__(self):
        return('GKDE(h='+str(self.h)+')')
    
    def fit(self, X, y=None):
        
        if self.standard_scaler:
            self._standard_scaler = StandardScaler()
            self._data = self._standard_scaler.fit_transform(X)
        else:
            self._data = X
            
        self._low_bounds = self._data[:, self.low_bounded_features].min(axis=0)
        self._high_bounds = self._data[:, self.high_bounded_features].max(axis=0)
        
        self._n = self._data.shape[0]
        self._d = self._data.shape[1]
        
        self._normalization = 1 / ((2*np.pi)**(self._d/2) * np.sqrt((self.h**2)**self._d))
        
        self._nn = NearestNeighbors(radius = self.h * self.support_factor,
                                   algorithm = self.algorithm,
                                   leaf_size = self.leaf_size,
                                   n_jobs = self.n_jobs)
        self._nn.fit(self._data)
        
        return(self)
        
    def predict(self, X):
        if self.standard_scaler:
            X = self._standard_scaler.transform(X)
        
        distances, _ = self._nn.radius_neighbors(X, radius=self.h * self.support_factor, return_distance=True)
        
        if self.n_jobs == 1:
            f = np.array([_gaussian(dist/self.h) for dist in distances])
        else:
            pool = Pool(self.n_jobs)
            f = pool.starmap(_gaussian, [(dist/self.h,) for dist in distances])
            f = np.array(f)
        f *= self._normalization / self._n
        
        # boundary bias correction
        for id_k, k in enumerate(self.low_bounded_features):
            f /= 1 / 2 * (1 + erf((X[:,k] - self._low_bounds[id_k]) / self.h / np.sqrt(2)))
                    
        for id_k, k in enumerate(self.high_bounded_features):
            f /= 1 / 2 * (1 + erf((self._high_bounds[id_k] - X[:,k]) / self.h / np.sqrt(2)))
        
        # boundary cutting if necessary
        f[np.any(X[:, self.low_bounded_features] < self._low_bounds, axis=1)] = 0
        f[np.any(X[:, self.high_bounded_features] > self._high_bounds, axis=1)] = 0
        
        if self.standard_scaler:
            f /= np.product(self._standard_scaler.scale_)
        
        return(f)

    def J(self):
        
        # on mutualise le calcul de distances.
        # on regarde donc d'abord dans un rayon de 3h puis on prend ce que l'on
        # veut ensuite.
        distances, _ = self._nn.radius_neighbors(self._data, radius=self.h * self.support_factor, return_distance=True)
        
        J = self._normalization * (np.sum([1 / self._n**2 / 2**(self._d/2) * np.sum(np.exp(-1 / 2 * self.h**2 * dist[dist<=self.h * self.support_factor / np.sqrt(2)]**2)) - 2 / (self._n -1) / self._n * np.sum(np.exp(-0.5 * dist**2 / self.h**2)) for dist in distances]) + 2 / (self._n-1) * np.exp(0))
        
        return(J)
    
    def marginal(self,
                 x,
                 k):
        
        X = np.zeros((x.size, self._d))
        X[:,k] = x
        
        if self.standard_scaler:
            X = self._standard_scaler.transform(X)
        x = X[:, k]
        
        nn = NearestNeighbors(radius=self.h * self.support_factor,
                              algorithm = self.algorithm,
                              leaf_size=self.leaf_size,
                              n_jobs = self.n_jobs)
        nn.fit(self._data[:,[k]])
        
        distances, _ = nn.radius_neighbors(x[:,None],
                                           radius= self.h * self.support_factor,
                                           return_distance=True)
        
        f = 1 / (2 * np.pi)**(1/2) / self.h / self._n * np.array([_gaussian(dist / self.h) for dist in distances])
        
        if k in self.low_bounded_features:
            id_k = self.low_bounded_features.index(k)
            f /= 1 / 2 * (1 + erf((x - self._low_bounds[id_k]) / self.h / np.sqrt(2)))
            
            f[x < self._low_bounds[id_k]] = 0
        
        if k in self.high_bounded_features:
            id_k = self.high_bounded_features.index(k)
            f /= 1 / 2 * (1 + erf((self._high_bounds[id_k] - x) / self.h / np.sqrt(2)))
            
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
            a = self._data[:,k].min() - self.support_factor * self.h
        
        if k in self.high_bounded_features:
            b = self._high_bounds[self.high_bounded_features.index(k)]
        else:
            b = self._data[:,k].max() + self.support_factor * self.h
        
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