#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:48:12 2020

@author: frem
"""

from imblearn.pipeline import Pipeline 

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from ..estimation import UnderSamplingEstimator

def make_calibrator(estimator=None,
                     under_sampler=None,
                     beta=None,
                     u=None,
                     calibration_method=None,
                     cv=5,
                     scaler=None,
                     discretizer=None):
    """
    make a calibrator as a scikit-learn Pipeline.

    Parameters
    ----------
    estimator : Estimator, required
        The estimator.
    under_sampler : Undersampler, default=None
        The under sampler. The sampling strategy is computed according beta and u.
        Only the majority class (u) is undersampled.
        If None, no resampling is made.
    beta : float in ]0,1], default=None.
        beta under sampling parameter. If None, beta is equal to max(N_v)/N_u where v neq u.
    u : integer, default=None
        The majority class. In LUCC, it corresponds to the initial state.
    calibration_method : 'sigmoid' or 'isotonic', default=None
        The method to use for calibration. Can be 'sigmoid' which
        corresponds to Platt's method (i.e. a logistic regression model) or
        'isotonic' which is a non-parametric approach. It is not advised to
        use isotonic calibration with too few calibration samples
        ``(<<1000)`` since it tends to overfit. If None, no calibration is made.
    cv : integer
        Determines the cross-validation StratifiedKFold splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
    scaler : Scaler, default=None
        The scaler. If None, no scaling is made.
    discretizer : Discretizer, default=None
        The discretizer. If None, no discretizing is made.
    feature_selector : FeatureSelector, default=None
        The feature selector. If None, no feature selection is made.
    
    Returns
    -------
    The calibrator (a scikit-learn Pipeline).

    """
        
    # estimator construction
    if calibration_method is not None:
        estimator = CalibratedClassifierCV(estimator,
                                           cv=StratifiedKFold(n_splits=cv,
                                                              shuffle=True),
                                           method=calibration_method)
    
    if under_sampler is not None:
        estimator = UnderSamplingEstimator(estimator=estimator,
                                           sampler=under_sampler,
                                           u=u)
        
    # pipeline construction
    pipe_list = []
    
    if scaler is not None:
        pipe_list.append(('scaler', scaler))
        
    if discretizer is not None:
        pipe_list.append(('discretizer', discretizer))
    
    pipe_list.append(('estimator', estimator))
    
    return(Pipeline(pipe_list))