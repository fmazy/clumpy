#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 12:10:40 2020

@author: frem
"""


from ._cluster import _Cluster

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors


class Agglomerative(_Cluster):
    def __init__(self, n_clusters=10, n_sample=100, affinity='euclidean', linkage='ward', distance_threshold=None):
        self.n_clusters = n_clusters
        self.n_sample = n_sample
        self.affinity = affinity
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        
        
    def _new_estimator(self):
        return(_AgglomerativeEstimator(n_clusters=self.n_clusters,
                                       n_sample = self.n_sample,
                                     affinity = self.affinity,
                                     linkage = self.linkage,
                                     distance_threshold = self.distance_threshold))
            
        
class _AgglomerativeEstimator():
    def __init__(self, n_clusters=10, n_sample=100, affinity='euclidean', linkage='ward', distance_threshold=None):
        self.n_clusters = n_clusters
        self.n_sample = n_sample
        self.affinity = affinity
        self.linkage = linkage
        self.distance_threshold = distance_threshold
    
    def fit(self, X):
        
        print('agglomerative clustering')
        ac = AgglomerativeClustering(n_clusters=self.n_clusters,
                                     affinity = self.affinity,
                                     linkage = self.linkage,
                                     distance_threshold = self.distance_threshold)
        
        self.X_sample = X[np.random.choice(np.arange(X.shape[0]),
                                       size=self.n_sample,
                                       replace=False),:]
        
        self.labels_sample = ac.fit_predict(self.X_sample)
        
        self.neigh = NearestNeighbors(n_neighbors=1)
        self.neigh.fit(self.X_sample)
            
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
        return(self.labels_sample[self.neigh.kneighbors(X, n_neighbors=1, return_distance=False)[:,0]].reshape(-1,1))
        
        
        