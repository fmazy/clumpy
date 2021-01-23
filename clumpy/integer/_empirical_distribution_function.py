#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 12:28:54 2021

@author: frem
"""

import numpy as np

def empirical_distribution_function(X):
    """
    Empirical distribution function of the samples ``X``.
    
    For now, works only for one-features values.
    
    .. math::
        F_n(x) = \cfrac{1}{n} \sum_{i=1}^n \mathbf{1}_{x_i \le x}
        
    where :math:`\mathbf{1}_{A}` is the indicator of the event :math:`A`.
    
    See `Wikipedia <https://en.wikipedia.org/wiki/Empirical_distribution_function>`_.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples to determine the empirical distribution function

    Returns
    -------
    F_n : array-like of shape (n_samples)
        The empirical distribution values for each samples.

    """
    if X.shape[1] > 1:
        raise(ValueError("The empirical distribution function works only for 1-dimensions for now."))
    
    X = X.copy()[:,0]
    X.sort()
    
    Fn = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        Fn[i] = (X<=X[i]).sum()
    Fn = Fn / X.shape[0]
    
    return(Fn)