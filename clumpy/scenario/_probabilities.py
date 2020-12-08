"""probabilities blabla"""

import numpy as np

def adjust_probabilities(P, f):
    """adjust probabilities

    Parameters
    ----------
    P : TYPE
        DESCRIPTION.
    f : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    if np.abs(f.sum() - 1) > 10**-10:
        raise(TypeError("Uncorrect scenario. The sum of frequencies should be equal to one."))
    
    
    P_mean = P.mean(axis=0)
    
    num = (f / P_mean)[None,:] * P
    den = ((f / P_mean)[None,:] * P).sum(axis=1)
    
    return(num/den[:,None])
        