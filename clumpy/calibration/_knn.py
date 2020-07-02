#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 10:05:07 2020

@author: frem
"""


import sklearn.svm
import sklearn.neighbors
import pandas as pd

#from ..allocation import build

class KNeighborsRegressor():
    def __init__(self, params):
        self.params = params
        
    def fit(self, J):
        self.transition = {}
        for trans, param in self.params.items():
            print(trans)
            
            P_vf__vi_z = build.computes_P_vf__vi_z(J)
            
            J_with_P = J.merge(right=P_vf__vi_z, how='left')
            J_with_P.fillna(0, inplace=True)
            
            X = J_with_P.loc[(J_with_P.v.i==trans[0])][['z']].values
            y = J_with_P.loc[(J_with_P.v.i==trans[0])][('P_vf__vi_z',trans[1])].values.T
            
            self.transition[trans] = sklearn.neighbors.KNeighborsRegressor(param['n_neighbors'], weights=param['weights']).fit(X, y)
            
    def predict(self, J):
        P = pd.DataFrame(index = J.index.values)
        for trans, param in self.params.items():
            X = J.loc[J.v.i==trans[0]][['z']].values
            
            P.loc[J.v.i==trans[0], str(trans[1])] = self.transition[trans].predict(X)
        
        return(P)
            
    def scores(self, J):
        P = pd.DataFrame(index = J.index.values)
        for trans, param in self.params.items():
            print(trans)
            
            P_vf__vi_z = build.computes_P_vf__vi_z(J)
            
            J_with_P = J.merge(right=P_vf__vi_z)
            
            X = J_with_P.loc[(J_with_P.v.i==trans[0])][['z']].values
            y = J_with_P.loc[J_with_P.v.i==trans[0]][('P_vf__vi_z',trans[1])].values.T
            
            print(self.transition[trans].score(X, y))
        
        return(P)