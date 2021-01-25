#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 12:28:54 2021

@author: frem
"""

import numpy as np
import itertools
from tqdm import tqdm

def empirical_distribution_function(X, verbose=0):
    """
    Empirical distribution function of the samples ``X``.
    
    For now, works only for one-features values.
    
    .. math::
        F_n(x) = \cfrac{1}{n} \sum_{i=1}^n \mathbf{1}_{x_i \le x}
        
    where :math:`\mathbf{1}_{A}` is the indicator of the event :math:`A`.
    
    See `Wikipedia <https://en.wikipedia.org/wiki/Empirical_distribution_function>`_.
    
    In multi-features cases, two (n, 2**n_features) numpy arrays are required
    to compute the EDF in each quadrants.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples to determine the empirical distribution function

    Returns
    -------
    F_n : array-like of shape (n_samples, 2**n_features)
        The empirical distribution values for each samples for each quadrants.

    """
    
    n = X.shape[0]
    n_features = X.shape[1]
    
    # B is the matrix of all permutations
    # 0 means the ~ operator, i.e. invert: ~True = False.
    B = np.array([list(x) for x in itertools.product((0, 1), repeat=n_features)])
    
    # Fn will be the EDF
    Fn = np.zeros((n, B.shape[0]))
    
    if verbose == 0:
        loop = range(n)
    elif verbose > 0:
        loop = tqdm(range(n))
    
    # for each element
    for i in loop:
        
        # L is the test : is the element <= to the row i
        # (test realized feature by feature)
        L = X <= X[i,:]
        
        # A is the matrix which create the sets
        # according to quadrant defined by B.
        A = np.zeros((n, B.shape[0])).astype(bool)
        
        # A is initialized : all elements belongs to all quadrants
        A.fill(True)
        
        # for each quadrants
        for id_F in range(B.shape[0]):
            # for each features
            for id_feature in range(B.shape[1]):
                # if this feature in this quadrant is marked as 0 in B
                # then, the test L should be inversed
                # its an inner operator &
                if B[id_F, id_feature] == 0:
                    A[:,id_F] = A[:,id_F] & ~L[:,id_feature]
                # else this feature in this quadrant is marked as 1 in B
                # then, the test L should be taken as it is
                # its an inner operator &
                else:
                    A[:,id_F] = A[:,id_F] & L[:,id_feature]
        
        # The EDF for each quadrants are equal to the number of elements...
        Fn[i,:] = A.sum(axis=0)    
    
    # ... divided by the total number of elements.
    Fn /= n
    
    return(Fn)