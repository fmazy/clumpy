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

    """
    def __init__(self,
                 density_estimator,
                 conditional_density_estimators,
                 n_corrections_max = 1000,
                 log_computations = False,
                 verbose = 0):
        
        self.density_estimator = density_estimator
        self.conditional_density_estimators = conditional_density_estimators
        self.n_corrections_max = n_corrections_max
        self.log_computations = log_computations
        self.verbose = verbose
        
        
    def fit(self, X, V):
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
                
        for state_v, cde in self.conditional_density_estimators.items():
            # select X_v
            X_v = X[V==state_v.value]
            
            # Density estimation fit
            cde.fit(X_v)
            
        return(self)
    
    def transition_probability(self, Y, P_v, palette_v):
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
        
        # forbid_null_value is forced to True by default for this density estimator
        self.density_estimator.set_params(forbid_null_value = True)
        
        self.density_estimator.fit(Y)
        
        # list_v_id is the list of v index except u.
        # list_v_id = list(np.arange(len(self.list_v)))
        # list_v_id.remove(self.list_v.index(self.u))
        
        # P(Y) estimation
        P_Y = self.density_estimator.predict(Y)[:,None]
        
        # P(Y|v) estimation
        P_Y__v = np.vstack([cde.predict(Y) for cde in self.conditional_density_estimators.values()]).T
        
        # compose P_v in the same order as self.conditional_density_estimators.keys()
        P_v = np.array([P_v[palette_v.get_id(state)] for state in self.conditional_density_estimators.keys()])
                
        # it should only remain the no-transited state
        if P_v.size != len(palette_v) - 1:
            raise(ValueError('conditional density estimators and palette_v are not compatible.'))
        
        # BAYES PROCESS
        if self.log_computations == False:
            # if no log computation
            P_v__Y = P_Y__v / P_Y
            P_v__Y *= P_v / P_v__Y.mean(axis=0)
            
            s = P_v__Y.sum(axis=1)
            
        else:
            # with log computation
            log_P_Y__v = np.zeros_like(P_Y__v)
            log_P_Y__v.fill(-np.inf)
            
            log_P_Y__v = np.log(P_Y__v, where=P_Y__v>0, out=log_P_Y__v)
            
            log_P_Y = np.log(P_Y)
            
            log_P_v__Y = log_P_Y__v - log_P_Y
            log_P_v__Y -= np.log(np.mean(np.exp(log_P_v__Y), axis=0))
            log_P_v__Y += np.log(P_v)
            
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
                    
                    P_v__Y *= P_v / P_v__Y.mean(axis=0)
                    
                    s = np.sum(P_v__Y, axis=1)
                else:
                    # with log computation
                    log_P_v__Y[id_anomalies] = log_P_v__Y[id_anomalies] - np.log(s[id_anomalies][:, None])
                    
                    log_P_v__Y -= np.log(np.mean(np.exp(log_P_v__Y), axis=0))
                    log_P_v__Y += np.log(P_v)
                    
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
        
        cde_states = list(self.conditional_density_estimators.keys())
        complete_P_v__Y = np.zeros((P_v__Y.shape[0], len(palette_v)))
        
        for id_state, state in enumerate(palette_v):
            if state in cde_states:
                complete_P_v__Y[:, id_state] = P_v__Y[:, cde_states.index(state)]
            else:
                complete_P_v__Y[:, id_state] = 1 - P_v__Y.sum(axis=1)
        
        return(complete_P_v__Y)