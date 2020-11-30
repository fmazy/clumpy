# -*- coding: utf-8 -*-

from ..utils import check_parameter_vi

import numpy as np
from copy import deepcopy

class Discretizer():
    def __init__(self, strategy='uniform', n_bins=None):
        self.strategy = strategy
        self.n_bins = n_bins
    
    def fit(self, X_u):
        check_parameter_vi(X_u)
        
        self.bins_u = {}
        
        for u in X_u.keys():
            X = X_u[u]
            
            for id_feature in range(X.shape[1]):

                if self.strategy == 'quantile':  # Determine bin edges by distribution of data
                    
                    quantiles = np.linspace(0, 1, self.n_bins[u][id_feature] + 1)
                    self.bins_u[(u, id_feature)] = np.quantile(X[:, id_feature], quantiles)
                    self.bins_u[(u, id_feature)][-1] = self.bins_u[(u, id_feature)][-1] + 1e-8
                    
                    # monotonic correction
                    # it is possible to have numeric approximation with negative difference (-10**-16)
                    # first get differences x_{i+1}-x_{i}
                    d = np.diff(self.bins_u[(u, id_feature)])
                    # append 1 at the end to have index i+1
                    d_before = np.append(d, 1)
                    # append 0 at the beginning to have index i
                    d_after = np.append(0, d)
                    # set the correction to get real 0 difference.
                    self.bins_u[(u, id_feature)][d_after<0] = self.bins_u[(u, id_feature)][d_before<0]
                    # another way could be to make positive every negative difference...
                    
                    
                elif self.strategy == 'uniform':
                    
                    self.bins_u[(u, id_feature)] = np.linspace(0.,
                                                                 X[:, id_feature].max() + 1e-8,
                                                                 self.n_bins[u][id_feature] + 1)
            
                else:
                    raise ValueError("Invalid entry to 'strategy' input. Strategy "
                                     "must be either 'quantile' or 'uniform'.")
                
                
        
    def transform(self, X_u, inplace=False):
        check_parameter_vi(X_u)
        
        if not inplace:
            X_u = deepcopy(X_u)
        
        for (u, id_feature) in self.bins_u.keys():
            X_u[u][:, id_feature] = np.digitize(X_u[u][:, id_feature], self.bins_u[(u, id_feature)]) + 1
        
        if not inplace:
            return(X_u)
  