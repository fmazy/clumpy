#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:40:42 2020

@author: frem
"""


import sklearn.neighbors
import sklearn.model_selection

import pandas as pd

class KernelDensity():
    def __init__(self, params):
        self.params = params
        
    def fit(self, J):
        self.transition = {}
        for trans, param in self.params.items():
            print(trans)
            
            X = J.loc[(J.v.i==trans[0]) & (J.v.f==trans[1])][['z']].values
            
            GridSearchCVParams = {'bandwidth': param['GridSearchCV_bandwidth']}
            
            grid = sklearn.model_selection.GridSearchCV(sklearn.neighbors.KernelDensity(kernel=param['kernel']),
                                                        GridSearchCVParams)
            grid.fit(X)
            
            self.transition[trans] = sklearn.neighbors.KernelDensity(kernel=param['kernel'], bandwidth=grid.best_params_['bandwidth']).fit(X)

            
    def score_samples(self, J):
        P = pd.DataFrame(index = J.index.values)
        for trans, param in self.params.items():
            X = J.loc[J.v.i==trans[0]][['z']].values
            
            P.loc[J.v.i==trans[0], str(trans[1])] = self.transition[trans].score_samples(X)
        
        return(P)
            