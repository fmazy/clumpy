#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 10:05:07 2020

@author: frem
"""
from ._calibration import _Calibration
from ..definition._case import Case

import numpy as np
import sklearn.svm
import sklearn.neighbors
import pandas as pd
from ._calibration import _clean_X

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
        
    # def feature_selection(self, case):
        
    
    def fit(self, P_z__vi_vf):
        """
        Fit the model using J as training data

        Parameters
        ----------
        J : pandas dataframe.
            A two level ``z`` (X) and ``P_vf__vi_z`` (y) columns are expected.

        """
        
        
        self.k_beighbors_classifiers = {}
        
        for vi in P_z__vi_vf.keys():
            self.k_beighbors_classifiers[vi] = sklearn.neighbors.KNeighborsRegressor(n_neighbors=self.n_neighbors,
                                                                                      weights=self.weights,
                                                                                      algorithm=self.algorithm,
                                                                                      leaf_size=self.leaf_size,
                                                                                      p=self.p,
                                                                                      metric=self.metric,
                                                                                      metric_params=self.metric_params,
                                                                                      n_jobs=self.n_jobs)

            self.k_beighbors_classifiers[vi].fit(P_z__vi_vf[vi].z.values, 
                                                 P_z__vi_vf[vi].P_z__vi_vf.values)
            
    def predict(self, case):
        P_z__vi_vf = {}
        for vi in case.Z.keys():
            
            # df = pd.DataFrame(case.Z[vi])
            # df_unique = df.drop_duplicates()
            
            # df_unique('P_z__vi_vf')
            P_z__vi_vf[vi] = self.k_beighbors_classifiers[vi].predict(case.Z[vi])
            
        return(P_z__vi_vf)
    
    # def predict(self, J):
    #     """
    #     Predict the target for the provided data.

    #     Parameters
    #     ----------
    #     J : pandas dataframe.
    #         A two level ``z`` feature column is expected.

    #     Returns
    #     -------
    #     J_predicted : pandas dataframe
    #         Target values above the ``P_vf__vi_z`` column.

    #     """
    #     J = J.reindex(sorted(J.columns), axis=1)
        
    #     J = J[[('v','i')]+J[['z']].columns.to_list()].copy()
        
    #     P_vf__vi_z_names = [('P_vf__vi_z', vf) for vf in self.list_vf]
    #     for P_vf__vi_z_name in P_vf__vi_z_names:
    #         J[P_vf__vi_z_name] = 0
        
    #     for vi in J.v.i.unique():
    #         X = J.loc[J.v.i == vi, 'z'].values
    #         X = _clean_X(X)
            
    #         P_vf__vi_z_names_without_vi = P_vf__vi_z_names.copy()
    #         P_vf__vi_z_names_without_vi.remove(('P_vf__vi_z', vi))
            
    #         J.loc[J.v.i == vi, P_vf__vi_z_names_without_vi] = self.k_beighbors_classifiers[vi].predict(X)
            
    #         J.loc[J.v.i == vi, ('P_vf__vi_z', vi)] = 1 - J.loc[J.v.i == vi].P_vf__vi_z.sum(axis=1)
                
    #     J = J.reindex(sorted(J.columns), axis=1)

    #     return(J)
    
    
    def score(self, J, y):
        """
        Return the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(). The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0. It returns the coefficient R^2 for each initial states.

        Parameters
        ----------
        J : pandas dataframe.
            A two level ``z`` feature column is expected.
        
        y : numpy array.
            True ``P_vf__vi_z`` values for J in the same order.

        Returns
        -------
        score : list of floats
            R^2 of self.predict(J) wrt. y for each ``vi`` in the ascending order.

        """
        J = J.reindex(sorted(J.columns), axis=1)
        J.reset_index(inplace=True)
        
        score = []
        for vi in np.sort(J.v.i.unique()):
            idx = J.loc[J.v.i == vi].index.values
            
            X = J.loc[idx, 'z'].values
            X = _clean_X(X) # remove NaN columns
            
            
            # focus on different final state
            list_vf = self.list_vf.copy()
            idx_vi = list_vf.index(vi)
            idx_vf = list(np.arange(len(list_vf)))
            idx_vf.remove(idx_vi)
                        
            yx = y[idx,:]
            yx = yx[:, idx_vf]
            
            score.append(self.k_beighbors_classifiers[vi].score(X, yx))
        
        return(score)
