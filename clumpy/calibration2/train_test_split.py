#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 12:01:02 2020

@author: frem
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def train_test_split_non_null_constraint(X, y, test_size=0.2):
    if len(y.shape) > 1:
        non_null_values = y.sum(axis=1) > 0
        
        X_train_nnv, X_test_nnv, y_train_nnv, y_test_nnv = train_test_split(X[non_null_values,:],
                                                                            y[non_null_values,:],
                                                                            test_size=test_size)
        X_train_nv, X_test_nv, y_train_nv, y_test_nv = train_test_split(X[~non_null_values,:],
                                                                        y[~non_null_values,:],
                                                                        test_size=test_size)
        
    else:
        non_null_values = y > 0
        
        X_train_nnv, X_test_nnv, y_train_nnv, y_test_nnv = train_test_split(X[non_null_values,:],
                                                                            y[non_null_values],
                                                                            test_size=test_size)
        X_train_nv, X_test_nv, y_train_nv, y_test_nv = train_test_split(X[~non_null_values,:],
                                                                        y[~non_null_values],
                                                                        test_size=test_size)
    
    
    
    X_train = np.concatenate((X_train_nnv, X_train_nv))
    X_test = np.concatenate((X_test_nnv, X_test_nv))
    y_train = np.concatenate((y_train_nnv, y_train_nv))
    y_test = np.concatenate((y_test_nnv, y_test_nv))
    
    return(X_train, X_test, y_train, y_test)