#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:37:01 2020

@author: frem
"""

from .._calibration import _Calibration, compute_P_z__vi_vf

class _Cluster():
    
    def fit(self, case):
        self.estimators = {}
        
        for vi in case.Z.keys():
            print('vi=',vi)
            
            self.estimators[vi] = self._new_estimator()
            self.estimators[vi].fit(case.Z[vi], case.vf[vi], list_vf=case.dict_vi_vf[vi])
            
    def predict(self, case):
        labels = {}
        P_z__vi_vf = {}
        
        for vi in case.Z.keys():
            labels[vi], P_z__vi_vf[vi] = self.estimators[vi].predict(case.Z[vi])
        
        return(labels, P_z__vi_vf)