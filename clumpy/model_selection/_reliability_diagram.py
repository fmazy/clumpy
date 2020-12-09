#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 09:22:13 2020

@author: frem
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from matplotlib import pyplot as plt

# from ..utils import check_list_parameters_vi
from sklearn.metrics import brier_score_loss as sklearn_brier_score_loss

def calibration_curve(y_true, y_prob, *, normalize=False, n_bins=5,
                      strategy='uniform'):
    """
    [Fork of sklearn function called "calibration_curve" https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html#sklearn.calibration.calibration_curve]
    
    Compute true and predicted probabilities for a calibration curve.

    The method assumes the inputs come from a binary classifier, and
    discretize the [0, 1] interval into bins.

    Calibration curves may also be referred to as reliability diagrams.

    The difference with sklearn resides in the max bound of the uniform
    linspace call which was set to `1` and is set to `y_prob.max()` here.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True targets.

    y_prob : array-like of shape (n_samples,)
        Probabilities of the positive class.

    normalize : bool, default=False
        Whether y_prob needs to be normalized into the [0, 1] interval, i.e.
        is not a proper probability. If True, the smallest value in y_prob
        is linearly mapped onto 0 and the largest one onto 1.

    n_bins : int, default=5
        Number of bins to discretize the [0, 1] interval. A bigger number
        requires more data. Bins with no samples (i.e. without
        corresponding values in `y_prob`) will not be returned, thus the
        returned arrays may have less than `n_bins` values.

    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy used to define the widths of the bins.

        uniform
            The bins have identical widths.
        quantile
            The bins have the same number of samples and depend on `y_prob`.

    Returns
    -------
    prob_true : ndarray of shape (n_bins,) or smaller
        The proportion of samples whose class is the positive class, in each
        bin (fraction of positives).

    prob_pred : ndarray of shape (n_bins,) or smaller
        The mean predicted probability in each bin.

    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).
    """
    y_true = y_true
    y_prob = y_prob

    if normalize:  # Normalize predicted values into interval [0, 1]
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    elif y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1] and normalize is "
                         "set to False.")

    if strategy == 'quantile':  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8
    elif strategy == 'uniform':
        bins = np.linspace(0., y_prob.max() + 1e-8, n_bins + 1)
    else:
        raise ValueError("Invalid entry to 'strategy' input. Strategy "
                         "must be either 'quantile' or 'uniform'.")
    
    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    return prob_true, prob_pred

def reliability_diagram(y_true, y_prob, n_bins=10, classes_to_hide=None, strategy='uniform'):
    plt.figure(figsize=(20,5))
    
    classes = np.unique(y_true)
    
    id_plt = 0
    for id_c,c in enumerate(classes):
        if c not in classes_to_hide:
            id_plt += 1
            plt.subplot(1, len(classes), id_plt)
                
            fraction_of_positives, mean_predicted_value = calibration_curve(np.array(y_true == c, int),
                                                                            y_prob[:,id_c],
                                                                            n_bins=n_bins,
                                                                            strategy=strategy)
            
            phi_BS = np.sum((fraction_of_positives-mean_predicted_value)**2)
            
            m = max([mean_predicted_value.max(), fraction_of_positives.max()])
        
            plt.plot([0, m], [0, m], "k:", label="Perfectly calibrated")
            plt.scatter(mean_predicted_value, fraction_of_positives)
            plt.ylabel('fraction of positives')
            plt.xlabel('mean predicted value')
            plt.title(" -> "+str(c)+", $\phi="+str(round(phi_BS,4))+'$')

def reliability_diagram3(y_true_vi,
                        y_prob_vi,
                        sigmoid_a=None,
                        sigmoid_b=None,
                        n_bins=10):
    
    check_list_parameters_vi([y_true_vi, y_prob_vi])
    
    
    # get n_classes for subplot init
    n_classes = 0
    classes_vi = {}
    for vi in y_true_vi.keys():
        classes_vi[vi] = np.unique(y_true_vi[vi])
        
        if len(classes_vi[vi]) > n_classes:
            n_classes = len(classes_vi[vi]) - 1
    
    id_vi = -1 # to get a each vi on each lines
    for vi in y_true_vi.keys():
        id_vi += 1 # to get a each vi on each lines
        
        y_true = y_true_vi[vi]
        y_prob = y_prob_vi[vi]
        classes = classes_vi[vi]
        
        id_plt = id_vi * n_classes # to get a each vi on each lines
        
        for id_c,c in enumerate(classes):
            if c != vi:
            
                id_plt += 1
                
                plt.subplot(len(y_true_vi), n_classes, id_plt)
                    
                fraction_of_positives, mean_predicted_value = calibration_curve(np.array(y_true == c, int),
                                                                                y_prob[:,id_c],
                                                                                n_bins=n_bins,
                                                                                strategy='uniform')
                
                m = max([mean_predicted_value.max(), fraction_of_positives.max()])
            
                plt.plot([0, m], [0, m], "k:", label="Perfectly calibrated")
                plt.scatter(mean_predicted_value, fraction_of_positives)
                plt.title(str(vi)+" -> "+str(c))
    
    

def reliability_diagram2(clf_vi,
                        X_train_vi, X_test_vi, y_train_vi, y_test_vi,
                        vi = None,
                        vf = None,
                        display_vi_equal_vf = False,
                        n_bins=10,
                        strategy='uniform',
                        display_sigmoid=False,
                        sigmoid_cv=2,
                        display_isotonic=False,
                        isotonic_cv=2):
    
    check_list_parameters_vi([clf_vi, X_train_vi, X_test_vi, y_train_vi, y_test_vi])
    
    if vi is None:
        vi_list = list(clf_vi.keys())
    else:
        vi_list = [vi]
    
    id_plt = 0
    
    for vi in vi_list:
        clf = clf_vi[vi]
        X_train = X_train_vi[vi]
        X_test = X_test_vi[vi]
        y_train = y_train_vi[vi]
        y_test = y_test_vi[vi]
        
        clf.fit(X_train, y_train)
        
        if hasattr(clf, "predict_proba"):
            predict_proba = clf.predict_proba(X_test)
        else:  # use decision function as in https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html#sphx-glr-auto-examples-calibration-plot-calibration-curve-py
            predict_proba = clf.decision_function(X_test)
            predict_proba = \
                (predict_proba - predict_proba.min()) / (predict_proba.max() - predict_proba.min())
        
        
        classes = clf.classes_
        list_vf = {}
        
        for id_c, c in enumerate(classes):
            if vf is not None:
                if c == vf:
                    list_vf[id_c] = c
            else:
                if not display_vi_equal_vf:
                    if vi != c:
                        list_vf[id_c] = c
                else:
                    list_vf[id_c] = c
        
        
        for id_c, c in list_vf.items():
            id_plt += 1
            plt.subplot(len(vi_list), len(list_vf), id_plt)
            
            fraction_of_positives, mean_predicted_value = calibration_curve(np.array(y_test == c, int),
                                                                            predict_proba[:,id_c],
                                                                            n_bins=n_bins,
                                                                            strategy=strategy)
            
            m = max([mean_predicted_value.max(), fraction_of_positives.max()])
        
            plt.plot([0, m], [0, m], "k:", label="Perfectly calibrated")
            plt.scatter(mean_predicted_value, fraction_of_positives)
            plt.title("vf="+str(c))
        
    return(True)
    
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, predict_proba, n_bins=n_bins, strategy=strategy)
    
    m = max([mean_predicted_value.max(), fraction_of_positives.max()])
    
    plt.plot([0, m], [0, m], "k:", label="Perfectly calibrated")
    plt.scatter(mean_predicted_value, fraction_of_positives)
    
    if display_sigmoid:
        
        sigmoid = CalibratedClassifierCV(clf, cv=sigmoid_cv, method='sigmoid')
        sigmoid.fit(X_train, y_train)
        
        a = sigmoid.calibrated_classifiers_[1].calibrators_[0].a_
        b = sigmoid.calibrated_classifiers_[1].calibrators_[0].b_
    
        x = np.linspace(0,m,100)
        p = 1/(1+np.exp(a*x+b))
        
        plt.plot(x,p)
        
    if display_isotonic:
        
        isotonic = CalibratedClassifierCV(clf, cv=isotonic_cv, method='isotonic')
        isotonic.fit(X_train, y_train)
        
        ir = isotonic.calibrated_classifiers_[1].calibrators_[0]
    
        x = np.linspace(0,m,100)
        p = ir.predict(x)
        
        plt.plot(x,p)
        
    e = np.average((y_test - predict_proba) ** 2)
    print(e)