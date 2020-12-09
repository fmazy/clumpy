"""log score blabla"""

import numpy as np
from sklearn.metrics import make_scorer

from ..resampling import under_sampling

def log_score(y_true, y_prob, a, b=1):
    """Log score
    
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
    
    See Also
    ---------
    :func:`clumpy.metrics.log_scorer`:
        To create a scorer.
    
    :func:`clumpy.metrics.compute_a`:
        To compute the ``a`` parameter.

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
    """Computes log score parameter ``a``.

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    a

    See Also
    ---------
    :func:`clumpy.metrics.log_score`:
        To compute the log score.
    
    :func:`clumpy.metrics.log_scorer`:
        To create a scorer.
    """
    n = y.shape[0]
    
    unique_y, ni = np.unique(y,return_counts=True)
    
    unique_y = list(unique_y)
    
    fi = ni/n
    
    return(-1 / np.sum(fi*np.log(fi)))
    
def log_scorer(a, b=1):
    """make log scorer.

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.

    Returns
    -------
    scorer
    
    See Also
    ---------
    :func:`clumpy.metrics.log_score`:
        To compute the log score.
    
    :func:`clumpy.metrics.compute_a`:
        To compute the ``a`` parameter.

    """
    return(make_scorer(score_func=log_score,
                        greater_is_better=True,
                        needs_proba=True,
                        a=a,
                        b=b))

def _undersampling_log_score(y_true, y_prob, beta, id_u, a, b=1):
    """
    undersampling log score

    Parameters
    ----------
    y_true : TYPE
        DESCRIPTION.
    y_prob : TYPE
        DESCRIPTION.
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
    return(log_score(y_true=y_true,
                     y_prob = under_sampling.correct_probabilities(y_prob, beta, id_u),
                     a=a,
                     b=b))

def under_sampling_log_scorer(beta, id_u, a, b=1):
    """log scorer
    

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
    return(make_scorer(score_func=_undersampling_log_score,
                        greater_is_better=True,
                        needs_proba=True,
                        beta=beta,
                        id_u=id_u,
                        a=a,
                        b=b))
    