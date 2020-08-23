#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 12:01:02 2020

@author: frem
"""

import pandas as pd
from sklearn.model_selection import train_test_split as sklearn_train_test_split


def train_test_split(J, test_size=0.25):
    """
    Train test split for each initial state.
    
    Parameters
        ----------
        J : pandas dataframe.
            A two level ``z`` column is expected.

        Returns
        -------
        splitting : list
            List containing train-test split of inputs.
    """
    J_train = pd.DataFrame()
    J_test  = pd.DataFrame()
    
    for vi in J.v.i.unique():
        J_vi_train, J_vi_test = sklearn_train_test_split(J.loc[J.v.i==vi], test_size=test_size)
        
        J_train = pd.concat([J_train, J_vi_train])
        J_test = pd.concat([J_test, J_vi_test])
        
    return(J_train, J_test)