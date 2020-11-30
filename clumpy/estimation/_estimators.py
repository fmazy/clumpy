#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:02:05 2020

@author: frem
"""

import numpy as np

from ..metrics import log_score
from ..utils import check_list_parameters_vi, check_parameter_vi

class Estimators():
    """Base probability estimator for a whole case.
    
    Parameters
    ----------
    clf_vi : dict of estimators
        Estimators of classification for each initial state vi. 
    """
    def __init__(self, clf_vi):
        
        check_parameter_vi(clf_vi)
        
        self.clf_vi = clf_vi
        self.classes_ = np.array(list(clf_vi.keys())).astype(int)
    
    def fit(self, X_vi, y_vi):
        """
        Fit the estimator according to Z_vi, y_vi
        
        X_vi : dict of array-likes of shape (n_samples, n_features)
            Training vectors for each initial state v_i, where n_samples is the number of samples
            and n_features is the number of features.
        
        y_vi : dict of array-likes of shape (n_samples,)
            Target values for each initial states v_i.
        """
        
        check_list_parameters_vi([X_vi, y_vi], list_vi = self.classes_)
        
        for vi, clf in self.clf_vi.items():
            clf.fit(X_vi[vi], y_vi[vi])
    
    def predict_proba(self, X_vi):
        """
        For each vi, return probability estimates for the test vector X.

        Parameters
        ----------
        X_vi : dict of array-likes of shape (n_samples, n_features)
            Test vectors for each initial state v_i, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        C : dict of array-likes of shape (n_samples, n_classes)
            For each initial state v_i, returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.

        """
        
        check_list_parameters_vi([X_vi], list_vi = self.classes_)
        
        C = {}
        for vi in self.classes_:
            
            # test if clf has predict_proba method
            if hasattr(self.clf_vi[vi], "predict_proba"):
                C[int(vi)] = self.clf_vi[vi].predict_proba(X_vi[vi])
            else:  # use decision function as in https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html#sphx-glr-auto-examples-calibration-plot-calibration-curve-py
                C[int(vi)] = self.clf_vi[vi].decision_function(X_vi[vi])
                C[int(vi)] = \
                    (C[int(vi)] - C[int(vi)].min()) / (C[int(vi)].max() - C[int(vi)].min())
        
        return(C)
    
    def score(self, X_vi, y_vi):
        """
        Return the evaluation metric score.

        X_vi : dict of array-likes of shape (n_samples, n_features)
            Test vectors for each initial state v_i, where n_samples is the number of samples
            and n_features is the number of features.
        
        y_vi : dict of array-likes of shape (n_samples,)
            Target values for each initial states v_i.
            
        method : {'brier', 'logloss'}, default='brier'
            evaluation metrics method
            
            brier
                THe brier score as in :meth:`metrics.brier_score_loss`
                
            logloss
                THe brier score as in :meth:`metrics.log_loss`

        Returns
        -------
        The score for each transition of each initial state vi.

        """
        
        check_list_parameters_vi([X_vi, y_vi], list_vi = self.classes_)
        
        y_pred = self.predict_proba(X_vi)
        
        
        return(log_score(y_vi, y_pred))
        
        
    
