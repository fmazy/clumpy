#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 10:44:36 2020

@author: frem
"""
import numpy as np

from sklearn.metrics import make_scorer

def log_score(y_true, y_prob, a, b=1):
    """
    Log score
    
    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True targets.
    y_prob : array, shape (n_samples,)
        Probabilities of the positive class.
    sample_weight : ignored for now.
    a : float
        log score parameter, computed with :func:`compute_a`.
    
    Returns
    -------
    score : float
        Log loss score
    """
    
    y_prob = y_prob.copy()
    
    n = y_prob.shape[0]
    
    unique_y_true, ni = np.unique(y_true,return_counts=True)
    
    unique_y_true = list(unique_y_true)
    
    i = np.zeros(y_true.size)
    
    for y in unique_y_true:
        i[y_true == y] = unique_y_true.index(y)
    i = i.astype(int)
    
    idx = np.column_stack((np.arange(i.size), i))
    
    # on donne aux probabilités nulles une petite chance si jamais ça a eu lieu effectivement
    # cette petite chance est égale à 0.01 de la plus petite chance
    # ça évite d'avoir un -inf en sortie du log...
    y_prob[y_prob <= 0] = y_prob[y_prob>0].min() * 0.01
    
    
    s = b + a/n*np.sum(np.log(y_prob[tuple(idx.T)]))
    
    return(s)

def compute_a(y):
    """
    Computes log score parameter ``a``.

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    a

    """
    n = y.shape[0]
    
    unique_y, ni = np.unique(y,return_counts=True)
    
    unique_y = list(unique_y)
    
    fi = ni/n
    
    return(-1 / np.sum(fi*np.log(fi)))
    
def log_scorer(a, b=1):
    """
    make log scorer.

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.

    Returns
    -------
    scorer

    """
    return(make_scorer(score_func=log_score,
                        greater_is_better=True,
                        needs_proba=True,
                        a=a,
                        b=b))