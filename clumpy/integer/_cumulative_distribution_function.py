#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 13:01:15 2021

@author: frem
"""

import ghalton
import numpy as np

def cumulative_distribution_function(f, X, method='quasi_monte_carlo', n=1000, a='min', b='max', args={}):
    """
    Computes cumulative distribution function through Monte Carlo method.

    Parameters
    ----------
    f : function
        A python function or method to integrate which returns an image of any `X`.
        
    X : array-like of shape (n_samples, n_features)
        The samples where the cumulative distribution function is computed.
    
    method : {'monte_carlo', 'quasi_monte_carlo'}, default='quasi_monte_carlo'
        The integration method. Possible values :
            
            - 'monte_carlo' : the `Monte Carlo integration <https://en.wikipedia.org/wiki/Monte_Carlo_integration>`_.
            - 'quasi_monte_carlo' : the `Quasi-Monte Carlo integration <https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_method>`_.
        
    n : int, default=1000.
        The number of monte carlo samples.
        
    a : {'min', 'min-var'} or array-like of shape (n_features), default='min'
        The lower limit of integration. Possible values :
            
            - 'min' : the minimum value of ``X``.
            - 'min-var' : the minimum value of ``X`` minus its variance.
            - [array-like of shape (n_features)] : a choosen value.
            
    b : {'max', 'max+var'} or array-like of shape (n_features), default='max'
        The upper limit of integration. Possible values :
            
            - 'max' : the maximum value of ``X``.
            - 'max+var' : the maximum value of ``X`` plus its variance.
            - [array-like of shape (n_features)] : a choosen value.
    
    args : dict, default={}
        extra arguments to pass to ``f``.

    Returns
    -------
    F : array-like of shape (n_samples)
        The cumulative distribution function values for each sample.

    """
    sequencer = ghalton.GeneralizedHalton(X.shape[1])
    X_halton = np.array(sequencer.get(n))
        
    X_halton = X_halton * (b-a) + a
    
    f_halton = f(X_halton)
    
    V = (b-a)**X.shape[1]
    
    r = np.zeros(X.shape[0])
    
    for i in range(X.shape[0]):
        
        b_r = X[i,:]
        
        id_X_r = X_halton[:,0] <= b_r
        r[i] = V / n * f_halton[id_X_r].sum()
    
    return(r)
    