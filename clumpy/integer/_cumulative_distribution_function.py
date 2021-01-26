#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 13:01:15 2021

@author: frem
"""

import ghalton
import numpy as np
from tqdm import tqdm
import itertools

def cumulative_distribution_function(f, X, method='quasi_monte_carlo', n_mc=1000, a='min', b='max', verbose=0, args={}):
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
        
    n_mc : int, default=1000.
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
    
    *kwargs : dict
        extra arguments to pass to ``f``.

    Returns
    -------
    F : array-like of shape (n_samples)
        The cumulative distribution function values for each sample.

    """
    
    
    if method == 'quasi_monte_carlo':
        sequencer = ghalton.GeneralizedHalton(X.shape[1])
        X_mc = np.array(sequencer.get(n_mc))
    elif method == 'monte_carlo':
        X_mc = np.random.random(size=n_mc)
    else:
        raise(ValueError("Invalid 'method' argument. Should be one of {'quasi_monte_carlo', 'monte_carlo'}."))
    
    if type(a) == str:
        if a == 'min':
            a = X.min(axis=0)
        elif a == 'min-var':
            a = X.min(axis=0) - X.var(axis=0)
    
    if type(b) == str:
        if b == 'max':
            b = X.max(axis=0)
        elif b == 'max+var':
            b = X.max(axis=0) + X.var(axis=0)
    
    # print(a)
    # print(b)
    
    n = X.shape[0]
    n_features = X.shape[1]
    
    # B is the matrix of all permutations
    # 0 means the ~ operator, i.e. invert: ~True = False.
    B = np.array([list(x) for x in itertools.product((0, 1), repeat=n_features)])
    
    X_mc = X_mc * (b-a) + a
    
    F_mc = np.vstack([f(X_mc, *args) for id_F in range(B.shape[0])]).T
    
    V = np.prod((b-a))
    # print(V)
    
    F = np.zeros((n, B.shape[0]))
    
    if verbose == 0:
        loop = range(n)
    elif verbose > 0:
        loop = tqdm(range(n))
    
    # for each element
    for i in loop:
        
        L = X_mc <= X[i,:]
        # A is the matrix which create the sets
        # according to quadrant defined by B.
        A = np.zeros((n_mc, B.shape[0])).astype(bool)
        
        # A is initialized : all elements belongs to all quadrants
        A.fill(True)
        
        
        # for each quadrants
        for id_F in range(B.shape[0]):
            # for each features
            for id_feature in range(B.shape[1]):
                # if this feature in this quadrant is marked as 1 in B
                # then, the test L should be inversed
                # its an inner operator &
                if B[id_F, id_feature] == 1:
                    A[:,id_F] = A[:,id_F] & ~L[:,id_feature]
                # else this feature in this quadrant is marked as 0 in B
                # then, the test L should be taken as it is
                # its an inner operator &
                else:
                    A[:,id_F] = A[:,id_F] & L[:,id_feature]
        
        F[i,:] = (F_mc*A).sum(axis=0)
    
    F *= V / n_mc
    
    return(F)
    