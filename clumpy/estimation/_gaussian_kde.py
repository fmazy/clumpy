#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 20:16:24 2021

@author: frem
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

class GKDE():
    def __init__(self, h, n_jobs=1):
        self.h = h
    
    def fit(self, X):
        self.data = X
        self.n = X.shape[0]
        self.d = X.shape[1]
        
        self.normalization = 1 / ((2*np.pi)**(self.d/2) * np.sqrt((self.h**2)**self.d))
                
        self.nn = NearestNeighbors(radius=self.h * 3,
                                   algorithm='kd_tree')
        self.nn.fit(X)
        
    def pdf(self, X):
        distances, _ = self.nn.radius_neighbors(X, radius=self.h * 3, return_distance=True)
        
        # f = 1 / self.n / self.gauss_integral_value / self.V / self.h**d * np.array([np.sum(np.exp(-0.5 * (d/self.h)**2)) for d in distances])
        
        f = np.array([np.sum(np.exp(-0.5 * (d/self.h)**2)) for d in distances]) * self.normalization / self.n
        
        return(f)

    def J(self):
        
        # on mutualise le calcul de distances.
        # on regarde donc d'abord dans un rayon de 3h puis on prend ce que l'on
        # veut ensuite.
        distances, _ = self.nn.radius_neighbors(self.data, radius=self.h * 3, return_distance=True)
        
        J = self.normalization * (np.sum([1 / self.n**2 / 2**(self.d/2) * np.sum(np.exp(-1 / 2 * self.h**2 * dist[dist<=self.h * 3 / np.sqrt(2)]**2)) - 2 / (self.n -1) / self.n * np.sum(np.exp(-0.5 * dist**2 / self.h**2)) for dist in distances]) + 2 / (self.n-1) * np.exp(0))
        
        return(J)