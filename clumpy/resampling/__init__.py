# -*- coding: utf-8 -*-

import numpy as np

def compute_sampling_strategy(y, beta=None, u=None):
    
    e_, n = np.unique(y, return_counts=True)
        
    if u is None:
        # on part du principe que le majoritaire est le non changement.
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
                sampling_strategy[e_i] =  n[e_!=u].max()
                beta = sampling_strategy[u] / n[id_u_]
            else:
                sampling_strategy[e_i] = n[i] * beta
    return(sampling_strategy)