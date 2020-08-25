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
    K-Nearest-Neighbors calibration
    """
    
    def __init__(self, n_neighbors=5, weights='distance', algorithm='auto'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
    
    def fit(self, J):
        """
        fit model with a discretized case

        Parameters
        ----------
        J : pandas dataframe.
            A two level ``z`` column is expected.

        """
        
        self.k_beighbors_classifiers = {}
        self.list_vf = list(np.sort(J.P_vf__vi_z.columns.to_list()))
        
        for vi in J.v.i.unique():
            X = J.loc[J.v.i==vi, 'z'].values
            
            
            list_vf_without_vi = self.list_vf.copy()
            list_vf_without_vi.remove(vi)
            y = J.loc[J.v.i==vi, [('P_vf__vi_z', vf) for vf in list_vf_without_vi]].values
            
            X = _clean_X(X) # remove NaN columns
            
            self.k_beighbors_classifiers[vi] = sklearn.neighbors.KNeighborsRegressor(n_neighbors=self.n_neighbors,
                                                                                      weights=self.weights,
                                                                                      algorithm=self.algorithm)

            self.k_beighbors_classifiers[vi].fit(X, y)
    
    def predict(self, J):
        # index_init = J.index.values
        # J = J.sort_values(('v','i'))
        
        J = J[[('v','i')]+J[['z']].columns.to_list()].copy()
        J.reset_index(drop=False, inplace=True)
        J_proba = pd.DataFrame()
        
        for vi in J.v.i.unique():
            print('vi',vi)
            
            list_vf_without_vi = self.list_vf.copy()
            list_vf_without_vi.remove(vi)
            
            cols = [('v','i')] + J[['z']].columns.to_list() + [('P_vf__vi_z', vf) for vf in list_vf_without_vi]
            cols = pd.MultiIndex.from_tuples(cols)
            J_proba_vi = pd.DataFrame(columns=cols)
            
            
            X = J.loc[J.v.i==vi, 'z'].values
            J_proba_vi[['z']] = X
            
            X = _clean_X(X) # remove NaN columns
            
            J_proba_vi['P_vf__vi_z'] = self.k_beighbors_classifiers[vi].predict(X)
            J_proba_vi[('v','i')] = vi
            
            J_proba_vi[('P_vf__vi_z', vi)] = 1 - J_proba_vi.P_vf__vi_z.sum(axis=1)
            
            J = J.merge(J_proba_vi, how='left')
        
        J.set_index('index', inplace=True)

        J = J.reindex(sorted(J.columns), axis=1)

        return(J)
    
    
    def score(self, J, y):
        
        J = J.copy()
        J.reset_index(inplace=True)
        
        s = []
        for vi in J.v.i.unique():
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
            
            s.append(self.k_beighbors_classifiers[vi].score(X, yx))
        
        return(s)
