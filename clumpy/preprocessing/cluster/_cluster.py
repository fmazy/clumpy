#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:37:01 2020

@author: frem
"""

class _Cluster():
    
    def fit(self, case):
        self.estimators = {}
        
        for vi in case.Z.keys():
            print('vi=',vi)
            
            self.estimators[vi] = self._new_estimator()
            self.estimators[vi].fit(case.Z[vi])
            
    def transform(self, case, inplace=False):
        if not inplace:
            case = case.copy()
        
        for vi in case.Z.keys():
            case.Z[vi] = self.estimators[vi].predict(case.Z[vi])
            case.Z_names[vi] = ['clustered']
        
        if not inplace:
            return(case)