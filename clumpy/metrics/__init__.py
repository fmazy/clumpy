#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:54:39 2020

@author: frem
"""

# from sklearn.metrics import brier_score_loss as sklearn_brier_score_loss
# from sklearn.metrics import log_loss as sklearn_log_loss
import numpy as np

from sklearn.metrics import make_scorer
# from copy import deepcopy

# from ..utils import check_list_parameters_vi

def log_score(y_true, y_prob):
    """
    Log score
    
    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True targets.
    y_prob : array, shape (n_samples,)
        Probabilities of the positive class.
    sample_weight : ignored for now.
    
    Returns
    -------
    score : float
        Log loss score
    """
    b = 1
    
    n = y_prob.shape[0]
    
    unique_y_true, ni = np.unique(y_true,return_counts=True)
    
    unique_y_true = list(unique_y_true)
    
    fi = ni/n
    
    a = -1 / np.sum(fi*np.log(fi))
    
    i = np.zeros(y_true.size)
    
    for y in unique_y_true:
        i[y_true == y] = unique_y_true.index(y)
    i = i.astype(int)
    
    idx = np.column_stack((np.arange(i.size), i))
    
    # on donne aux probabilités nulles une petite chance si jamais ça a eu lieu effectivement
    # cette petite chance est égale à 0.01 de la plus petite chance
    # ça évite d'avoir un -inf en sortie du log...
    y_prob[y_prob == 0] = y_prob[y_prob>0].min() * 0.01
    
    s = b+a/n*np.sum(np.log(y_prob[tuple(idx.T)]))
    
    return(s)

log_scorer = make_scorer(score_func=log_score,
                    greater_is_better=True,
                    needs_proba=True)

# def log_scorer():
#     return(make_scorer(score_func=log_score,
#                     greater_is_better=True,
#                     needs_proba=True))

# def brier_score_loss(y_true_vi, y_pred_vi):
#     """
#     Compute the Brier score. The smaller the Brier score, the better, hence the naming with "loss".
    
#     Parameters
#     ----------
#     y_true_vi : dict of array-likes or label indicator matrix
#         Ground truth (correct) labels for n_samples samples.
#     y_pred_vi : dict of array-likes of float, shape = (n_samples, n_classes) or (n_samples,)
#         Predicted probabilities, as returned by a classifier's
#         predict_proba method for each vi. The labels in ``y_pred_vi[vi]`` are assumed to be
#         ordered alphabetically, as done by :method:`_Estimator.predict_proba` or
#         :class:`_Calibrator.predict_proba`.
#     """
    
#     check_list_parameters_vi([y_true_vi, y_pred_vi])
    
#     phi_BS = {}
    
#     for vi in y_true_vi.keys():
#         classes = np.unique(y_true_vi[vi])
        
#         phi_BS[vi] = np.zeros(classes.size)
        
#         for i, vf in enumerate(classes):  
#             phi_BS[vi][i] = sklearn_brier_score_loss(y_true = np.array(y_true_vi[vi] == vf, int),
#                                                      y_prob = y_pred_vi[vi][:, i])
        
#     return(phi_BS)

# def log_score_u(v_u, P_u):
#     s = {}
    
#     P_u = deepcopy(P_u)
    
#     check_list_parameters_vi([v_u, P_u])
    
#     for u in v_u.keys():
#         b = 1
#         n = P_u[u].shape[0]
        
#         unique_v, ni = np.unique(v_u[u],return_counts=True)
        
#         unique_v = list(unique_v)
        
#         fi = ni/n
        
#         a = -1 / np.sum(fi*np.log(fi))
        
#         i = np.zeros(v_u[u].size)
#         for v in unique_v:
#             i[v_u[u] == v] = unique_v.index(v)
#         i = i.astype(int)
        
#         idx = np.column_stack((np.arange(i.size), i))
        
#         # on donne aux probabilités nulles une petite chance si jamais ça a eu lieu effectivement
#         # cette petite chance est égale à 0.01 de la plus petite chance
#         P_u[u][P_u[u] == 0] = P_u[u][P_u[u]>0].min() * 0.01
        
#         s[u] = b+a/n*np.sum(np.log(P_u[u][tuple(idx.T)]))
        
#     return(s)



# def log_loss(y_true_vi, y_pred_vi):
#     """Log loss, aka logistic loss or cross-entropy loss.
    
#     Parameters
#     ----------
#     y_true_vi : dict of array-likes or label indicator matrix
#         Ground truth (correct) labels for n_samples samples.
#     y_pred_vi : dict of array-likes of float, shape = (n_samples, n_classes) or (n_samples,)
#         Predicted probabilities, as returned by a classifier's
#         predict_proba method for each vi. The labels in ``y_pred_vi[vi]`` are assumed to be
#         ordered alphabetically, as done by :method:`_Estimator.predict_proba` or
#         :class:`_Calibrator.predict_proba`.
#     """
    
#     check_list_parameters_vi([y_true_vi, y_pred_vi])
    
#     phi_LL = {}
    
#     for vi in y_true_vi.keys():
#         classes = np.unique(y_true_vi[vi])
        
#         phi_LL[vi] = np.zeros(classes.size)
        
#         for i, vf in enumerate(classes):    
#             phi_LL[vi][i] = sklearn_log_loss(y_true = np.array(y_true_vi[vi] == vf, int),
#                                              y_pred = y_pred_vi[vi][:, i])
        
#     return(phi_LL)
        