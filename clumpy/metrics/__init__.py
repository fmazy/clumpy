#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:54:39 2020

@author: frem
"""

from sklearn.metrics import brier_score_loss as sklearn_brier_score_loss
from sklearn.metrics import log_loss as sklearn_log_loss
import numpy as np

from ..utils import check_list_parameters_vi

def brier_score_loss(y_true_vi, y_pred_vi):
    """
    Compute the Brier score. The smaller the Brier score, the better, hence the naming with "loss".
    
    Parameters
    ----------
    y_true_vi : dict of array-likes or label indicator matrix
        Ground truth (correct) labels for n_samples samples.
    y_pred_vi : dict of array-likes of float, shape = (n_samples, n_classes) or (n_samples,)
        Predicted probabilities, as returned by a classifier's
        predict_proba method for each vi. The labels in ``y_pred_vi[vi]`` are assumed to be
        ordered alphabetically, as done by :method:`_Estimator.predict_proba` or
        :class:`_Calibrator.predict_proba`.
    """
    
    check_list_parameters_vi([y_true_vi, y_pred_vi])
    
    phi_BS = {}
    
    for vi in y_true_vi.keys():
        final_classes = np.unique(y_true_vi[vi])
        
        phi_BS[vi] = np.zeros(final_classes.size)
        
        for i, vf in enumerate(final_classes):    
            phi_BS[vi][i] = sklearn_brier_score_loss(y_true = np.array(y_true_vi[vi] == vf, int),
                                                     y_prob = y_pred_vi[vi][:, i])
        
    return(phi_BS)

def log_loss(y_true_vi, y_pred_vi):
    """Log loss, aka logistic loss or cross-entropy loss.
    
    Parameters
    ----------
    y_true_vi : dict of array-likes or label indicator matrix
        Ground truth (correct) labels for n_samples samples.
    y_pred_vi : dict of array-likes of float, shape = (n_samples, n_classes) or (n_samples,)
        Predicted probabilities, as returned by a classifier's
        predict_proba method for each vi. The labels in ``y_pred_vi[vi]`` are assumed to be
        ordered alphabetically, as done by :method:`_Estimator.predict_proba` or
        :class:`_Calibrator.predict_proba`.
    """
    
    check_list_parameters_vi([y_true_vi, y_pred_vi])
    
    phi_LL = {}
    
    for vi in y_true_vi.keys():
        final_classes = np.unique(y_true_vi[vi])
        
        phi_LL[vi] = np.zeros(final_classes.size)
        
        for i, vf in enumerate(final_classes):    
            phi_LL[vi][i] = sklearn_log_loss(y_true = np.array(y_true_vi[vi] == vf, int),
                                             y_pred = y_pred_vi[vi][:, i])
        
    return(phi_LL)
        