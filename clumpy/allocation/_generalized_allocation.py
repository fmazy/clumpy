#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 12:25:38 2021

@author: frem
"""

import numpy as np

def generalized_allocation_rejection_test(P, list_v):
    """
    Generalized allocation rejection process.

    Parameters
    ----------
    P : array-like of shape (n_samples, n_classes)
        ``P.sum(axis=1)`` should be a vector of ones.
    list_v : list of int
        List of final classes. The classes should be in the same order as ``P`` classes.

    Returns
    -------
    y : array-like of shape (n_samples)
        The allocated classes.

    """
    P = np.nan_to_num(P)
    
    P[P<0] = 0
    
    y = np.zeros(P.shape[0])
    # cum sum along axis
    cs = np.cumsum(P, axis=1)
    
    # random value
    x = np.random.random(P.shape[0])
                            
    for id_vf in range(P.shape[1]):
        inv_id_vf = P.shape[1] - 1 - id_vf
        try:
            y[x < cs[:, inv_id_vf]] = list_v[inv_id_vf]
        except ValueError:
            print(x)
            print(cs[:, inv_id_vf])
            print(list_v)
            print(inv_id_vf)
        
    return(y)