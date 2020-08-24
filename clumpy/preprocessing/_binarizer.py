# -*- coding: utf-8 -*-

from .. import definition

import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from optbinning import MulticlassOptimalBinning

class Binarizer():
    def fit(self, J, fit_params, sound=0, plot=False):
        
        self.alpha = {}
    
        for (vi, feature_name), fit_param in fit_params.items():        
            X = J.loc[J.v.i==vi, ('z',feature_name)].values
            y = J.loc[J.v.i==vi, ('v', 'f')].values
            
            # if param['method'] == 'numpy':
            #     alpha_sub.alpha = _compute_bins_with_numpy(case.J, Zk, param['bins'])
                
            if fit_param['method'] == 'optbinning':
                self.alpha[(vi, feature_name)] = _compute_bins_with_optbinning(X, y, sound=sound, plot=plot)
                
            elif fit_param['method'] == 'numpy':
                if 'bins' not in fit_param.keys():
                    fit_param['bins'] = 'auto'
                self.alpha[(vi, feature_name)] = _compute_linear_bins(X, fit_param['bins'], sound=sound, plot=plot)
    
    def transform(self, J):
        J = J.copy()
        
        for (vi, feature_name), alpha in self.alpha.items():                    
            J.loc[J.v.i == vi, ('z', feature_name)] = np.digitize(J.loc[J.v.i == vi, ('z', feature_name)],
                                                                           bins=alpha)
        
        return(J)
        # fill na with 0
        # J.fillna(value=0, inplace=True)
    
    def inverse_transform(self, J, where='mean'):
        """
        Inverse the binarization.

        Parameters
        ----------
        J : pandas Dataframe
            Data to inverse. ``z`` 0-level column is expected.
        where : {'mean', 'right', 'left'} (default=``mean``)
            Where the value should be computed.

        """
        J = J.copy()
        
        for (vi, feature_name), alpha in self.alpha.items():
            
            if where=='mean':
                J.loc[(J.v.i == vi) &
                      (J.z[feature_name] != 0) &
                      (J.z[feature_name] != alpha.size), ('z', feature_name)] = (alpha[J.loc[J.v.i == vi, ('z', feature_name)].values.astype(int)-1] + alpha[J.loc[J.v.i == vi, ('z', feature_name)].values.astype(int)]) / 2
                
            elif where == 'left':
                J.loc[(J.v.i == vi) &
                          (J.z[feature_name] != 0) &
                          (J.z[feature_name] != alpha.size), ('z', feature_name)] = alpha[J.loc[J.v.i == vi, ('z', feature_name)].values.astype(int)-1]
                                                                                     
            elif where == 'right':
                J.loc[(J.v.i == vi) &
                          (J.z[feature_name] != 0) &
                          (J.z[feature_name] != alpha.size), ('z', feature_name)] = alpha[J.loc[J.v.i == vi, ('z', feature_name)].values.astype(int)]
                
            J.loc[(J.v.i == vi) &
                  (J.z[feature_name] == 0), ('z', feature_name)] = - np.inf
            
            J.loc[(J.v.i == vi) &
                  (J.z[feature_name] == alpha.size), ('z', feature_name)] = np.inf
        
        return(J)

def _compute_bins_with_optbinning(X, y, sound=0, plot=False):
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
        if plot:
            binning_table.plot()
        
        if sound >= 2:
            binning_table.build()
            binning_table.analysis()
    
    alpha = np.array([X.min()] + list(optb.splits) + [X.max()*1.0001])
    
    return(alpha)

def _compute_linear_bins(X, bins, sound=0, plot=False):
    alpha_N, alpha = np.histogram(X, bins=bins)
    # # on agrandie d'un chouilla le dernier bin pour inclure le bord supérieur
    alpha[-1] += (alpha[-1]-alpha[-2])*0.001
    
    if plot:
        plt.step(alpha, np.append(alpha_N,0), where='post')
        plt.show()
    
    return(alpha)