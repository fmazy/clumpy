#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:33:20 2020

@author: frem
"""

import numpy as np

from ._estimator import BaseEstimator

from ..resampling.under_sampling import compute_sampling_strategy, correct_probabilities

class UnderSamplingEstimator(BaseEstimator):
    """
    Resampling Estimator for classification probabilities
    """
    
    def __init__(self, estimator=None, sampler=None, u=None, sampling_strategy_method='max_N_v'):
        self.estimator = estimator
        self.sampler = sampler
        self.u = u
        self.sampling_strategy_method = sampling_strategy_method
        
    def fit(self, X, y):
        """Fit the model using X as training data and y as target values
        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training data. If array or matrix, shape [n_samples, n_features],
            or [n_samples, n_samples] if metric='precomputed'.
        v : {array-like, sparse matrix}
            Target values of shape = [n_samples] or [n_samples, n_outputs]
        """
        
        # under sampling initialization
        self.e_, n = np.unique(y, return_counts=True)
        
        if self.u is None:
            # on part du principe que le majoritaire est le non changement.
            self.id_u_ = list(n).index(n.max())
            self.u = self.e_[self.id_u_]
        else:
            self.id_u_ = list(self.e_).index(self.u)
            
        # sampling strategy
        if self.sampling_strategy_method == 'max_N_v':
            self.sampler.sampling_strategy = compute_sampling_strategy(y, u=self.u)
        
        # get beta
        self.beta = self.sampler.sampling_strategy[self.u] / n[self.id_u_]
        
        # under sampling
        X_res, y_res = self.sampler.fit_resample(X, y)
        
        
        
        # estimation
        self.estimator.fit(X_res, y_res)        
        
    def predict_proba(self, X):
        """Return probability estimates for the test data X.
        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.
        Returns
        -------
        p : ndarray of shape (n_queries, n_classes), or a list of n_outputs
            of such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        P = self.estimator.predict_proba(X)
        
        # under sampling correction
        P = probability_correction(P, self.beta, self.id_u_)
        # first columns where e_i != u
        # P[:, self.e_!=self.u] = P[:, self.e_!= self.u] * self.beta / ( self.beta + ( 1 - self.beta ) * P[:,self.id_u_][:,None] )
        
        # then the closure condition
        # P[:, self.id_u_] = 1 - P[:, self.e_!=self.u].sum(axis=1)
        
        return(P)