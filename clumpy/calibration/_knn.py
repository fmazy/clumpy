#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 10:05:07 2020

@author: frem
"""
from ._calibration import _Calibration
from ..definition._case import Case
from ._calibration import compute_P_z__vi_vf

import numpy as np
import sklearn.svm
import sklearn.neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm


#from ..allocation import build

class KNeighborsRegressor(_Calibration):
    """
    K-Nearest-Neighbors calibration.
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for kneighbors queries.
        
    weights : {'uniform', 'distance'} or callable, default='uniform'
        weight function used in prediction.  Possible values:
        
        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.
        
        Uniform weights are used by default.
        
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:
        
        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.
        
        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.
        
    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.
    
    p : int, default=2
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    
    metric : str or callable, default='minkowski'
        the distance metric to use for the tree.  The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric. See the documentation of :class:`DistanceMetric` for a
        list of available metrics. If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a sparse graph,
        in which case only "nonzero" elements may be considered neighbors.
    
    metric_params : dict, default=None
        Additional keyword arguments for the metric function.
    
    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. Doesn't affect :meth:`fit` method.
    """
    
    def __init__(self, n_neighbors=5,
                 weights='distance',
                 algorithm='auto',
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 metric_params=None,
                 n_jobs=None):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        
    def _new_estimator(self):
        return(sklearn.neighbors.KNeighborsRegressor(n_neighbors=self.n_neighbors,
                                                        weights=self.weights,
                                                        algorithm=self.algorithm,
                                                        leaf_size=self.leaf_size,
                                                        p=self.p,
                                                        metric=self.metric,
                                                        metric_params=self.metric_params,
                                                        n_jobs=self.n_jobs))
        
    
            
    
