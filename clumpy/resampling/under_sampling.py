"""Under Sampling"""

import numpy as np

def compute_sampling_strategy(y, gamma=1, beta=None, u=None, return_beta=False):
    """ Compute sampling strategy
    

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    gamma : TYPE, optional
        DESCRIPTION. The default is 1.
    beta : TYPE, optional
        DESCRIPTION. The default is None.
    u : TYPE, optional
        DESCRIPTION. The default is None.
    return_beta : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
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

def correct_probabilities(P, beta, id_u):
    """Correct probabilities estimated through an undersampling.

    Parameters
    ----------
    P : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.
    id_u : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    P = P.copy()
    
    id_e = np.arange(P.shape[1])
    
    P[:, id_e!=id_u] = P[:, id_e!=id_u] * beta / ( beta + ( 1 - beta ) * P[:,id_u][:,None] )
    
    # then the closure condition
    P[:, id_u] = 1 - P[:, id_e!=id_u].sum(axis=1)
    
    return(P)
    
