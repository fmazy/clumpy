#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 20:16:24 2021

@author: frem
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

class GKDE():
    def __init__(self, h):
        self.h = h
    
    def fit(self, X):
        self.data = X
        self.n = X.shape[0]
        self.d = X.shape[1]
        
        self.normalization = 1 / ((2*np.pi)**(self.d/2) * np.sqrt((self.h**2)**self.d))
        
        # self.V = volume_unit_ball(self.d)
        # self.gauss_integral_value = self.d * gauss_integral(self.d - 1)
        
        self.nn = NearestNeighbors(radius=self.h * 3,
                                   algorithm='kd_tree')
        self.nn.fit(X)
        
    def pdf(self, X):
        distances, _ = self.nn.radius_neighbors(X, return_distance=True)
        
        # f = 1 / self.n / self.gauss_integral_value / self.V / self.h**d * np.array([np.sum(np.exp(-0.5 * (d/self.h)**2)) for d in distances])
        
        f = np.array([np.sum(np.exp(-0.5 * (d/self.h)**2)) for d in distances]) * self.normalization / self.n
        
        return(f)
    
    # def integral_square(self):
    #     distances, _ = self.nn.radius_neighbors(self.data, radius=self.h * 3 / np.sqrt(2), return_distance=True)
        
    #     A = np.sum([np.sum(np.exp(-1 / 4 * self.h**2 * dist**2)) for dist in distances])
        
    #     A = A / self.n**2 / (4*np.pi)**(self.d/2) / self.h**self.d
        
    #     return(A)
    
    # def leave_one_out(self):
    #     loo = self.n / (self.n -1) * np.sum(self.pdf(self.data))
        
    #     loo = loo - self.n / (self.n-1) * self.normalization * np.exp(0)
        
    #     return(loo)
    
    # def J(self):
    #     return(self.integral_square() - 2 / self.n * self.leave_one_out())

    def J(self):
        
        # on mutualise le calcul de distances.
        # on regarde donc d'abord dans un rayon de 3h puis on prend ce que l'on
        # veut ensuite.
        distances, _ = self.nn.radius_neighbors(self.data, radius=self.h * 3, return_distance=True)
        
        integral_square = np.sum([np.sum(np.exp(-1 / 4 * self.h**2 * dist[dist<=self.h * 3 / np.sqrt(2)]**2)) for dist in distances])
        
        integral_square = integral_square / self.n**2 / (4*np.pi)**(self.d/2) / self.h**self.d
        
        loo = 1 / (self.n -1) / self.n * self.normalization * np.sum([np.sum(np.exp(-0.5 * dist**2 / self.h**2)) for dist in distances])
        
        loo = loo - 1 / (self.n-1) * self.normalization * np.exp(0)
        
        return(integral_square - 2 * loo)