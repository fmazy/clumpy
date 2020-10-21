#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 11:41:24 2020

@author: frem
"""

from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
import numpy as np

class StandardScaler():
    """
    Standardize ``z`` features by removing the mean and scaling to unit variance for each initial states.
    """
    
    def fit(self, case):
        """
        Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        J : pandas dataframe.
            A two level ``z`` column is expected.

        """
        self._standard_scalers_according_to_vi = {}
        
        for vi in case.J.keys():
            self._standard_scalers_according_to_vi[vi] = SklearnStandardScaler()  # doctest: +SKIP
            d = case.Z[vi]
            if len(case.Z[vi].shape) == 1:
                d = d.reshape(-1,1)
            self._standard_scalers_according_to_vi[vi].fit(d)  # doctest: +SKIP
            
    def transform(self, case, inplace=False):
        """
        Perform standardization by centering and scaling

        Parameters
        ----------
        J : pandas dataframe.
            A two level ``z`` column is expected.

        Returns
        -------
        Standardized J.

        """
        if not inplace:
            case = case.copy()
        
        for vi in case.Z.keys():
            d = case.Z[vi]
            if len(case.Z[vi].shape) == 1:
                d = d.reshape(-1,1)
            case.Z[vi] = self._standard_scalers_according_to_vi[vi].transform(d)
        
        if not inplace:
            return(case)
    
    def inverse_transform(self, case, inplace=False):
        """
        Scale back the data to the original representation

        Parameters
        ----------
        J : pandas dataframe.
            A two level ``z`` column is expected.

        Returns
        -------
        Standardized J.

        """
        if not inplace:
            case = case.copy()
        
        for vi in case.Z.keys():
            case.Z[vi] = self._standard_scalers_according_to_vi[vi].inverse_transform(case.Z[vi])
            
        if not inplace:
            return(case)