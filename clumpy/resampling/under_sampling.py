#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 10:17:22 2020

@author: frem
"""

from ..metrics import log_score

from sklearn.metrics import make_scorer
import numpy as np

def compute_sampling_strategy(y, gamma=1, beta=None, u=None, return_beta=False):
    
    e_, n = np.unique(y, return_counts=True)
        
    if u is None:
        # on part du principe que u est le majoritaire.
        id_u_ = list(n).index(n.max())
        u = e_[id_u_]
    else:
        id_u_ = list(e_).index(u)
    
    sampling_strategy = {}
    for i, e_i in enumerate(e_):
        if e_i != u:
            # si e_i est différent de u, on prend tous les éléments.
            # (nécessaire pour appliquer la formule !)
            sampling_strategy[e_i] = n[i]
        else:
            if beta is None:
                sampling_strategy[e_i] =  int(n[e_!=u].max() * gamma)
                beta = sampling_strategy[u] / n[id_u_]
            else:
                sampling_strategy[e_i] = n[i] * beta
    
    if return_beta:
        return(sampling_strategy, beta)
    else:
        return(sampling_strategy)

def probability_correction(P, beta, id_u):
    
    P = P.copy()
    
    id_e = np.arange(P.shape[1])
    
    P[:, id_e!=id_u] = P[:, id_e!=id_u] * beta / ( beta + ( 1 - beta ) * P[:,id_u][:,None] )
    
    # then the closure condition
    P[:, id_u] = 1 - P[:, id_e!=id_u].sum(axis=1)
    
    return(P)
    
def log_score_corrected(y_true, y_prob, beta, id_u, a, b=1):
    
    return(log_score(y_true=y_true,
                     y_prob = probability_correction(y_prob, beta, id_u),
                     a=a,
                     b=b))

def log_scorer_corrected(beta, id_u, a, b=1):
    """
    

    Parameters
    ----------
    beta : TYPE
        DESCRIPTION.
    id_u : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.
    b : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.

    """
    return(make_scorer(score_func=log_score_corrected,
                        greater_is_better=True,
                        needs_proba=True,
                        beta=beta,
                        id_u=id_u,
                        a=a,
                        b=b))
    