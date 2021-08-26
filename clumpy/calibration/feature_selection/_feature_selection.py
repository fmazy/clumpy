#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 22:16:44 2021

@author: frem
"""
import numpy as np
import pandas as pd

class _FeatureSelector():
    def get_support(self):
        return(self._cols_support)
    
    def transform_case(self, case):
        for u in case.params.keys():
            features = []
            
            for i, f in enumerate(case.params[u]['features']):
                if i in self._cols_support[u]:
                    features.append(f)
            
            case.params[u]['features'] = features
            
        return(case)

class VarianceThreshold(_FeatureSelector):
    def __init__(self, threshold = 0.3):
        self.threshold = threshold
    
    def fit(self, X_u):
        self._cols_support = {}
        
        for u, X in X_u.items():
            self._cols_support[u] = np.where(X.var(axis=0) >= self.threshold)[0]

class CorrelationThreshold(_FeatureSelector):
    def __init__(self, threshold=0.7):
        self.threshold = threshold
    
    def fit(self, X_u):
        self._cols_support = {}
        
        for u, X in X_u.items():
            df = pd.DataFrame(X)
            
            corr = df.corr().values
            
            selected_features = list(np.arange(corr.shape[0]))
            
            corr_tril = np.abs(corr)
            corr_tril = np.tril(corr_tril) - np.diag(np.ones(corr_tril.shape[0]))
            
            pairs = np.where(corr_tril>self.threshold)
            
            features_pairs = [(pairs[0][i], pairs[1][i]) for i in range(pairs[0].size)]
            
            excluded_features = []
            
            for f0, f1 in features_pairs:
                f0_mean = np.abs(corr[:,f0]).mean()
                f1_mean = np.abs(corr[:,f1]).mean()
                
                if f0_mean >= f1_mean:
                    feature_to_remove = f0
                    
                else:
                    feature_to_remove = f1
                
                excluded_features.append(feature_to_remove)
                selected_features.remove(feature_to_remove)
                
                # toutes les paires concernées sont retirées
                for g0, g1 in features_pairs:
                    if g0 == feature_to_remove or g1 == feature_to_remove:
                        features_pairs.remove((g0, g1))
            
            self._cols_support[u] = selected_features
    
    
        

