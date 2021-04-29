#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 08:21:59 2021

@author: frem
"""
import numbers
from KDEpy.BaseKDE import BaseKDE
from KDEpy.kernel_funcs import volume_unit_ball
from sklearn.neighbors import NearestNeighbors
import numpy as np
import functools

class BoxKDE(BaseKDE):
    def __init__(self, kernel=None, bw=1, norm=2):
        self.norm = norm
        kernel = 'box'
        super().__init__(kernel, bw)
        assert isinstance(self.norm, numbers.Number) and self.norm > 0
    
    def fit(self, data, weights=None):
        super().fit(data, weights)
        
        self._nn = NearestNeighbors(radius=self.bw/2, p=2)
        self._nn.fit(self.data)
        
        return(self)
        
    def evaluate(self, grid_points):
        super().evaluate(grid_points)
        
        return self._evalate_return_logic(self.predict(self.grid_points), self.grid_points)
    
    def predict(self, X):
        neigh_ind = self._nn.radius_neighbors(X, return_distance=False)
        
        volume = volume_unit_ball(d=self._nn.n_features_in_, p=self.norm)
        
        if self.weights is None:
            return(np.array([ni.size for ni in neigh_ind]) / self._nn.n_samples_fit_ / ((self.bw/2)**self._nn.n_features_in_ * volume))
        else:
            return(np.array([self.weights[ni].sum() for ni in neigh_ind]) / self._nn.n_samples_fit_ / ((self.bw/2)**self._nn.n_features_in_ * volume))