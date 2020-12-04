#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:32:25 2020

@author: frem
"""

from ._cluster import _Cluster

import numpy as np
from sklearn.neighbors import NearestNeighbors


class RandomSeeds(_Cluster):
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        
        
    def _new_estimator(self):
        return(_RandomSeedsEstimator(n_clusters=self.n_clusters))
            
        
class _RandomSeedsEstimator():
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
    
    def fit(self, X):
        self.X_seeds = X[np.random.choice(np.arange(X.shape[0]),
                                       size=self.n_clusters,
                                       replace=False),:]
        self.labels = np.arange(self.n_clusters)
        
        self.neigh = NearestNeighbors(n_neighbors=1)
        self.neigh.fit(self.X_seeds)
            
    def predict(self, X):
        """
        predict :math:`P(z|v_i,v_f)`

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        return_labels : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        return(self.labels[self.neigh.kneighbors(X, n_neighbors=1, return_distance=False)[:,0]].reshape(-1,1))
        
        
        