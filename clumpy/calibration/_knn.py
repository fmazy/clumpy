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
        self.list_vf = J.P_vf__vi_z.columns.to_list()
        
        for vi in J.v.i.unique():
            X = J.loc[J.v.i==vi, 'z'].values
            y = J.loc[J.v.i==vi, 'P_vf__vi_z'].values
            
            X = _clean_X(X) # remove NaN columns
            
            self.k_beighbors_classifiers[vi] = sklearn.neighbors.KNeighborsRegressor(n_neighbors=self.n_neighbors,
                                                                                      weights=self.weights,
                                                                                      algorithm=self.algorithm)

            self.k_beighbors_classifiers[vi].fit(X, y)
    
    def predict(self, J):
        J_proba = pd.DataFrame()
        
        for vi in J.v.i.unique():
            print('vi',vi)
            cols = [('v','i')] + J[['z']].columns.to_list() + [('P_vf__vi_z', vf) for vf in np.sort(self.list_vf)]
            cols = pd.MultiIndex.from_tuples(cols)
            J_proba_vi = pd.DataFrame(columns=cols)
            
            
            X = J.loc[J.v.i==vi, 'z'].values
            J_proba_vi[['z']] = X
            
            X = _clean_X(X) # remove NaN columns
            
            J_proba_vi['P_vf__vi_z'] = self.k_beighbors_classifiers[vi].predict(X)
            J_proba_vi[('v','i')] = vi
            J_proba = pd.concat([J_proba, J_proba_vi])
        
        return(J_proba)
    
    def kneighbors(self, vi, x, n):
        kneighbors = pd.DataFrame()
        
        return(self.k_beighbors_classifiers[vi].kneighbors(x, n))
            
    
    def predict_proba(self, J):
        
        J_proba = pd.DataFrame()
        
        for vi in J.v.i.unique():
            print('vi',vi)
            cols = [('P_vf__vi_z', vf) for vf in np.sort(self.list_vf_according_to_vi[vi])]+[('j','')]
            cols = pd.MultiIndex.from_tuples(cols)
            J_proba_vi = pd.DataFrame(columns=cols)
            
            X = J.loc[J.v.i==vi, 'z'].values
            X = _clean_X(X) # remove NaN columns
            
            # print(self.k_beighbors_classifiers[vi].predict_proba(X).shape)
            # print(J_proba_vi.loc[J.v.i==vi].shape)
            J_proba_vi[('j','')] = J.loc[J.v.i==vi].index.values
            J_proba_vi['P_vf__vi_z'] = self.k_beighbors_classifiers[vi].predict(X)
            # print(J_proba_vi.shape)
        
            J_proba = pd.concat([J_proba, J_proba_vi])
        
        J_proba.set_index('j', inplace=True)
        
        return(J_proba)

# class KNeighborsRegressor():
#     def __init__(self, params):
#         self.params = params
        
#     def fit(self, J):
#         self.transition = {}
#         for trans, param in self.params.items():
#             print(trans)
            
#             P_vf__vi_z = build.computes_P_vf__vi_z(J)
            
#             J_with_P = J.merge(right=P_vf__vi_z, how='left')
#             J_with_P.fillna(0, inplace=True)
            
#             X = J_with_P.loc[(J_with_P.v.i==trans[0])][['z']].values
#             y = J_with_P.loc[(J_with_P.v.i==trans[0])][('P_vf__vi_z',trans[1])].values.T
            
#             self.transition[trans] = sklearn.neighbors.KNeighborsRegressor(param['n_neighbors'], weights=param['weights']).fit(X, y)
            
#     def predict(self, J):
#         P = pd.DataFrame(index = J.index.values)
#         for trans, param in self.params.items():
#             X = J.loc[J.v.i==trans[0]][['z']].values
            
#             P.loc[J.v.i==trans[0], str(trans[1])] = self.transition[trans].predict(X)
        
#         return(P)
            
#     def scores(self, J):
#         P = pd.DataFrame(index = J.index.values)
#         for trans, param in self.params.items():
#             print(trans)
            
#             P_vf__vi_z = build.computes_P_vf__vi_z(J)
            
#             J_with_P = J.merge(right=P_vf__vi_z)
            
#             X = J_with_P.loc[(J_with_P.v.i==trans[0])][['z']].values
#             y = J_with_P.loc[J_with_P.v.i==trans[0]][('P_vf__vi_z',trans[1])].values.T
            
#             print(self.transition[trans].score(X, y))
        
#         return(P)