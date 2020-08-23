# -*- coding: utf-8 -*-

from .. import definition

import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from optbinning import MulticlassOptimalBinning

class Binarizer():
    def fit(self, J, fit_params):
        
        self.alpha = {}
    
        for (vi, feature_name), fit_param in fit_params.items():        
            X = J.loc[J.v.i==vi, ('z',feature_name)].values
            y = J.loc[J.v.i==vi, ('v', 'f')].values
            
            # if param['method'] == 'numpy':
            #     alpha_sub.alpha = _compute_bins_with_numpy(case.J, Zk, param['bins'])
                
            if fit_param['method'] == 'optbinning':
                self.alpha[(vi, feature_name)] = _compute_bins_with_optbinning(X, y)
    
    def transform(self, J):
        J = J.copy()
        
        for (vi, feature_name), alpha in self.alpha.items():                    
            J.loc[J.v.i == vi, ('z', feature_name)] = np.digitize(J.loc[J.v.i == vi, ('z', feature_name)],
                                                                           bins=alpha)
        
        return(J)
        # fill na with 0
        # J.fillna(value=0, inplace=True)

def _compute_bins_with_optbinning(X, y, sound=0, plot=0):
    optb = MulticlassOptimalBinning(name='Zk', solver="cp")
    
    optb.fit(X, y)
    
    if optb.status != 'OPTIMAL':
        sound = 2
    if sound >= 1 or plot==1:
        print(optb.status)
        binning_table = optb.binning_table
        
        
        # print(binning_table.quality_score())
        if sound == 1:
            print(binning_table.build())
        if plot == 1:
            binning_table.plot()
        
        if sound >= 2:
            binning_table.build()
            binning_table.analysis()
    
    alpha = np.array([X.min()] + list(optb.splits) + [X.max()*1.0001])
    
    return(alpha)