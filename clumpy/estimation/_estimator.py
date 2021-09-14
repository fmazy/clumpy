#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:03:33 2021

@author: frem
"""

from ..density_estimation import GKDE

import numpy as np
from matplotlib import pyplot as plt

class TransitionProbabilityEstimator():
    """
    Transition probability estimator.

    Parameters
    ----------
    de_P_x__v : dict of density estimators with v as keys.
        Estimators of :math:`P(Y|u,v)` for each studied final states :math:`v`
        based on observed data :math:`X`.
        If None, a new object with default parameters is created for each :math:`v`.
    
    de_P_y : density estimator
        Estimator of :math:`P(Y|u)`.
        
    default_density_estimation : DensityEstimator, default=GKDE
        Density estimator used by default if de_P_x__v or de_P_y are None.
    
    n_corrections_max : int, default=100
        Maximum number of corrections for :math:`P(v|Y)` adjustment according to :math:`P(v)`.
    
    log_computations : bool, default=False,
        If ``True``, logarithm is used to compute :math:`P(v|Y)`.
    
    verbose : int, default=0
        Verbosity level.

    """
    def __init__(self,
                 de_P_x__v = None,
                 de_P_y = None,
                 default_density_estimator = GKDE,
                 n_corrections_max = 100,
                 log_computations = False,
                 verbose = 0):
        
        self.de_P_x__v = de_P_x__v
        self.de_P_y = de_P_y
        self.default_density_estimator = default_density_estimator
        self.n_corrections_max = n_corrections_max
        self.log_computations = log_computations
        self.verbose = verbose
        
        
    def fit(self, X, V, u):
        """
        Fit the transition probability estimators. Only :math:`P(Y|u,v)` is
        concerned.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Observed features data.
        V : array-like of shape (n_samples,) of int type
            The final land use state values. Only studied final v should appear.
        u : int
            The initial land use state.

        Returns
        -------
        self : TransitionProbabilityEstimator
            The fitted transition probability estimator object.

        """
        self.u = u
        
        self.list_v = list(np.sort(np.unique(V)))
        
        # initialize density estimator for each v (except u)
        if self.de_P_x__v is None:
            self.de_P_x__v = {}
            for v in self.list_v:
                if v != self.u:
                    self.de_P_x__v[v] = self.default_density_estimator()
        
        for v, de_P_x__v in self.de_P_x__v.items():
            # select X_v
            X_v = X[V==v]
            
            # Density estimation fit
            de_P_x__v.fit(X_v)
            
        return(self)
    
    def transition_probability(self, Y, P_v):
        """
        Estimates transition probability.

        Parameters
        ----------
        Y : array-like of shape (n_samples, n_features)
            Samples to estimate transition probabilities
        P_v : array-like of shape (n_transitions,)
            Global transition probabilities ordered as ``self.list_v``, i.e.
            the numerical order of studied final land us states.

        Returns
        -------
        P_v__Y : ndarray of shape (n_samples, n_transitions)
            Transition probabilities for each studied final land use states ordered
            as ``self.list_v``, i.e. the numerical order of studied final land us states.
        """
        # P(Y) density estimation fitting
        if self.de_P_y is None:
            self.de_P_y = self.default_density_estimator()
        
        # forbid_null_value is forced to True by default for this density estimator
        self.de_P_y.set_params(forbid_null_value = True)
        
        self.de_P_y.fit(Y)
        
        # list_v_id is the list of v index except u.
        list_v_id = list(np.arange(len(self.list_v)))
        list_v_id.remove(self.list_v.index(self.u))
        
        # P(Y) estimation
        P_Y = self.de_P_y.predict(Y)[:,None]
        
        # P(Y|v) estimation
        P_Y__v = np.vstack([de_P_x__v.predict(Y) for de_P_x__v in self.de_P_x__v.values()]).T
        
        # Bayes process
        if self.log_computations == False:
            # if no log computation
            P_v__Y = P_Y__v / P_Y
            P_v__Y *= P_v[list_v_id] / P_v__Y.mean(axis=0)
            
            s = P_v__Y.sum(axis=1)
            
        else:
            # with log computation
            log_P_Y__v = np.zeros_like(P_Y__v)
            log_P_Y__v.fill(-np.inf)
            
            log_P_Y__v = np.log(P_Y__v, where=P_Y__v>0, out=log_P_Y__v)
            
            log_P_Y = np.log(P_Y)
            
            log_P_v__Y = log_P_Y__v - log_P_Y
            log_P_v__Y -= np.log(np.mean(np.exp(log_P_v__Y), axis=0))
            log_P_v__Y += np.log(P_v[list_v_id])
            
            s = np.sum(np.exp(log_P_v__Y), axis=1)
    
        if np.sum(s > 1) > 0:
            if self.verbose > 0:
                print('Warning, uncorrect probabilities have been detected.')
                print('Some global probabilities may be to high.')
                print('For now, some corrections are made.')
    
            n_corrections = 0
            
            while np.sum(s > 1) > 0 and n_corrections < self.n_corrections_max:
                id_anomalies = s > 1
                
                if self.log_computations == False:
                    # if no log computation
                    P_v__Y[id_anomalies] = P_v__Y[id_anomalies] / \
                        s[id_anomalies][:, None]
                    
                    P_v__Y *= P_v[list_v_id] / P_v__Y.mean(axis=0)
                    
                    s = np.sum(P_v__Y, axis=1)
                else:
                    # with log computation
                    log_P_v__Y[id_anomalies] = log_P_v__Y[id_anomalies] - np.log(s[id_anomalies][:, None])
                    
                    log_P_v__Y -= np.log(np.mean(np.exp(log_P_v__Y), axis=0))
                    log_P_v__Y += np.log(P_v[list_v_id])
                    
                    s = np.sum(np.exp(log_P_v__Y), axis=1)
                    
                n_corrections += 1
    
            if self.verbose > 0:
                print('Corrections done in '+str(n_corrections)+' iterations.')
            
            if n_corrections == self.n_corrections_max:
                print('Warning : the P(v|Y) adjustment algorithm has reached the maximum number of loops. The n_corrections_max parameter should be increased.')
        
        if self.log_computations:
            P_v__Y = np.exp(log_P_v__Y)
        
        # last control to ensure s <= 1
        id_anomalies = s > 1
        P_v__Y[id_anomalies] = P_v__Y[id_anomalies] / \
            s[id_anomalies][:, None]
        
        # avoid nan values
        P_v__Y = np.nan_to_num(P_v__Y)
        
        # create an array with the non transited u state column
        P_v__Y = np.insert(arr = P_v__Y,
                            obj = self.list_v.index(self.u),
                            values = 1 - P_v__Y.sum(axis=1),
                            axis=1)
        
        return(P_v__Y)