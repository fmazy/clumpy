#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:50:20 2020

@author: frem
"""


from ._cluster import _Cluster

from sklearn.cluster import Birch as BirchSklearn
import numpy as np

class Birch(_Cluster):
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        
    def _new_estimator(self):
        return(BirchSklearn(n_clusters = self.n_clusters))