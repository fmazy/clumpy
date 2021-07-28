# -*- coding: utf-8 -*-

def scotts_rule(X):
    """
    Scott's rule according to "Multivariate density estimation", Scott 2015, p.164.

    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features).

    Returns
    -------
    h : float
        Scotts rule bandwidth. The returned bandwidth should be then factored
        by the data variance.

    """
    n = X.shape[0]
    d = X.shape[1]
    return(n**(-1/(d + 4)))