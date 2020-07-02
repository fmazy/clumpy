#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 13:57:22 2020

@author: frem
"""

import numpy as np
import pandas as pd
import sklearn.feature_selection
import itertools
import scipy.stats

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LassoCV


def pearson_correlation(J, vi):
    X = J.loc[J.v.i == vi].z
    y = J.loc[J.v.i == vi].v.f.values
    
    df_cor = pd.DataFrame(columns=['Zk_name','pearson_correlation'])
    # calculate the correlation with y for each feature
    for i in X.columns.to_list():
        cor = abs(np.corrcoef(X[i], y)[0, 1])
        df_cor.loc[df_cor.index.size] = [i, cor]
    
    df_cor.fillna(0, inplace=True)
    df_cor.sort_values(by='pearson_correlation', ascending=False, inplace=True)
    
    print(df_cor)
    return(df_cor)

def chi2(J, vi):
    """

    Parameters
    ----------
    J : TYPE
        J is discretized
    vi : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    J = J.fillna(0, inplace=False)
    X = J.loc[J.v.i == vi].z
    y = J.loc[J.v.i == vi].v.f.values
    
    c = sklearn.feature_selection.chi2(X,y)
    
    df_chi2 = pd.DataFrame(columns=['Zk_name','chi2', 'pval'])
    df_chi2.Zk_name = X.columns.to_list()
    df_chi2.chi2 = c[0]
    df_chi2.pval = c[1]
    
    df_chi2.fillna(0, inplace=True)
    df_chi2.sort_values(by='chi2', ascending=False, inplace=True)
    
    print(df_chi2)
    return(df_chi2)

    
def random_forest(J, vi):
    J = J.fillna(0, inplace=False)
    X = J.loc[J.v.i == vi].z
    y = J.loc[J.v.i == vi].v.f.values
    
    forest = ExtraTreesClassifier(n_estimators=10,
                                  random_state=0)
    
    forest.fit(X, y)
    importances = forest.feature_importances_
    
    df_trees = pd.DataFrame(columns=['Zk_name','importance'])
    df_trees.Zk_name = X.columns.to_list()
    df_trees.importance = importances
    
    df_trees.sort_values(by='importance', ascending=False, inplace=True)
    
    print(df_trees)
    return(df_trees)

def lasso(J, vi):
    J = J.fillna(0, inplace=False)
    X = J.loc[J.v.i == vi].z
    y = J.loc[J.v.i == vi].v.f.values
    
    clf = LassoCV().fit(X, y)
    importances = np.abs(clf.coef_)
    
    df_lasso = pd.DataFrame(columns=['Zk_name','importance'])
    df_lasso.Zk_name = X.columns.to_list()
    df_lasso.importance = importances
    
    df_lasso.sort_values(by='importance', ascending=False, inplace=True)
    
    print(df_lasso)
    return(df_lasso)

def chi2_features_independence(J, vi):
    J = J.fillna(0, inplace=False)
    X = J.loc[J.v.i == vi].z
    
    z_couples = list(itertools.combinations(X.columns.to_list(),2))
    
    df_chi2 = pd.DataFrame(np.zeros((len(X.columns.to_list()),len(X.columns.to_list()))),columns=X.columns.to_list())
    
    df_chi2['z'] = X.columns.to_list()
    df_chi2.set_index('z', inplace=True)
    
    for couple in z_couples:
        df_O_ij = J.groupby(by=[('z',couple[0]), ('z', couple[1])]).size().reset_index(name='O_ij')
        size= (df_O_ij[('z',couple[0])].max(), df_O_ij[('z',couple[1])].max())
        
        O_ij = np.zeros(size)
        O_ij[df_O_ij[('z',couple[0])].values-1, df_O_ij[('z',couple[1])]-1] = df_O_ij.O_ij.values
        
        # print(O_ij)
        c = scipy.stats.chi2_contingency(O_ij)
        df_chi2.loc[couple[0], couple[1]] = c[1]
        df_chi2.loc[couple[1], couple[0]] = c[1]
    
    return(df_chi2)
