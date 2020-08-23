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
    
    def fit(self, J):
        """
        Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        J : pandas dataframe.
            A two level ``z`` column is expected.

        """
        self._standard_scalers_according_to_vi = {}
        
        for vi in J.v.i.unique():
            J_vi = J.loc[J.v.i==vi,'z'].values.copy()
            # check if a column is full of nan:
            columns_sumed_na = np.isnan(J_vi).sum(axis=0)
            for idx, c in enumerate(columns_sumed_na):
                if c == J_vi.shape[0]:
                    J_vi[:,idx] = 0
            
            self._standard_scalers_according_to_vi[vi] = SklearnStandardScaler()  # doctest: +SKIP
            self._standard_scalers_according_to_vi[vi].fit(J_vi)  # doctest: +SKIP
            
    def transform(self, J):
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
        J = J.copy()
        for vi in J.v.i.unique():
            J.loc[J.v.i==vi, 'z'] = self._standard_scalers_according_to_vi[vi].transform(J.loc[J.v.i==vi, 'z'].values)
        
        return(J)
    
    def inverse_transform(self, J):
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
        J = J.copy()
        for vi in J.v.i.unique():
            J.loc[J.v.i==vi, 'z'] = self._standard_scalers_according_to_vi[vi].inverse_transform(J.loc[J.v.i==vi, 'z'].values)
            
        return(J)