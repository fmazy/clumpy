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
    
    def fit(self, X, vf, list_vf=None):
        if list_vf is None:
            list_vf = np.unique(vf)
        
        print('agglomerative clustering')
        ac = AgglomerativeClustering(n_clusters=self.n_clusters,
                                     affinity = self.affinity,
                                     linkage = self.linkage,
                                     distance_threshold = self.distance_threshold)
        
        self.X_sample = X[np.random.choice(np.arange(X.shape[0]),
                                       size=self.n_sample,
                                       replace=False),:]
        
        self.labels_sample = ac.fit_predict(self.X_sample)
        
        print('P(z|v_i,v_f computing')
        self.neigh = NearestNeighbors(n_neighbors=1)
        self.neigh.fit(self.X_sample)
        
        labels = self.labels_sample[self.neigh.kneighbors(X, n_neighbors=1, return_distance=False)[:,0]]
        labels = pd.DataFrame(labels, columns=['labels'])
        
        print(labels.shape)
        
        self.P_z__vi_vf = np.zeros((self.n_clusters, len(list_vf)))
        
        for id_vfx, vfx in enumerate(list_vf):
            N_z__vi_vf = labels.loc[vf==vfx].groupby(by='labels').size().reset_index(name='n')
            N_z__vi_vf.n /= N_z__vi_vf.n.sum()
            
            self.P_z__vi_vf[N_z__vi_vf.labels.values ,id_vfx] = N_z__vi_vf.n.values / N_z__vi_vf.n.values.sum()
    
    def predict(self, X, return_labels=True):
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
        labels = self.labels_sample[self.neigh.kneighbors(X, n_neighbors=1, return_distance=False)[:,0]]
        labels_unique = np.unique(labels)
        
        if labels_unique.size != self.P_z__vi_vf.shape[0]:
            P_z__vi_vf_to_remove = self.P_z__vi_vf[np.isin(labels, labels_unique),:].sum(axis=0)
            P_z__vi_vf = self.P_z__vi_vf[labels,:] / (1-P_z__vi_vf_to_remove)
            
        else:
            P_z__vi_vf = self.P_z__vi_vf[labels,:]
        
        if return_labels:
            return(labels, P_z__vi_vf)
        else:
            return(P_z__vi_vf)
        
        