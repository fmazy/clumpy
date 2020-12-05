# -*- coding: utf-8 -*-

from ..utils import check_parameter_vi

import numpy as np
import warnings
from copy import deepcopy

class Discretizer():
    def __init__(self, strategy='quantile', n_bins=None, null_bin_width=10**(-15)):
        self.strategy = strategy
        self.n_bins = n_bins
        self.null_bin_width = null_bin_width
        
    def fit(self, X, y=None):
        """

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y
            Ignored
        """
        self.bins_ = []
        
        for id_feature in range(X.shape[1]):

            if self.strategy == 'quantile':  # Determine bin edges by distribution of data
                
                quantiles = np.linspace(0, 1, self.n_bins[id_feature] + 1)
                self.bins_.append(np.quantile(X[:, id_feature], quantiles))
                
                # remove 0 width bins
                # first get differences x_{i+1}-x_{i}
                d = np.diff(self.bins_[id_feature])
                # append 0 at the beginning to have index i
                d = np.append(1, d)
                # set the correction to get real 0 difference.
                self.bins_[id_feature] = np.delete(self.bins_[id_feature], d<self.null_bin_width)
                
                if self.bins_[id_feature].size != self.n_bins[id_feature] + 1:
                     warnings.warn('for the feature #'+str(id_feature)+', the required bin number is unreached. required: '+str(self.n_bins[id_feature])+', output: '+str(self.bins_[id_feature].size-1), 
                                stacklevel=2)
                
                # bounds are set to infinity
                self.bins_[id_feature][0] = -np.inf
                self.bins_[id_feature][-1] = np.inf
                
                
            elif self.strategy == 'uniform':
                
                self.bins_.append(np.linspace(0.,
                                                    X[:, id_feature].max() + 1e-8,
                                                    self.n_bins[id_feature] + 1))
                
                self.bins_[id_feature][0] = -np.inf
                self.bins_[id_feature][-1] = np.inf
        
            else:
                raise ValueError("Invalid entry to 'strategy' input. Strategy "
                                 "must be either 'quantile' or 'uniform'.")
                
    def transform(self, X, copy=False):
        """Perform standardization by centering and scaling

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: False)
            Copy the input X or not.
        """
        
        if not copy:
            X = deepcopy(X)
        
        for id_feature in range(X.shape[1]):
            X[:, id_feature] = np.digitize(X[:, id_feature], self.bins_[id_feature])
        
        if not copy:
            return(X)

