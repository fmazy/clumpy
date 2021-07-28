# -*- coding: utf-8 -*-

import numpy as np

class _WhiteningTransformer():
    def fit(self, X):
        self._mean = X.mean(axis=0)
        
        self._num_obs = X.shape[0]
        
        _, self._s, Vt = np.linalg.svd(X - self._mean, full_matrices=False)
        self._V = Vt.T
        
        self._transform_matrix = self._V @ np.diag(1 / self._s) * np.sqrt(self._num_obs-1)
        self._inverse_transform_matrix = np.diag(self._s)  @ self._V.T / np.sqrt(self._num_obs-1)
        
        self._transform_det = np.abs(np.linalg.det(self._transform_matrix))
        self._inverse_transform_det = np.abs(np.linalg.det(self._inverse_transform_matrix))
        
        return(self)
        
    def transform(self, X):
        X = X - self._mean
        return(X @ self._transform_matrix)
    
    def inverse_transform(self, X):
        X = X @ self._inverse_transform_matrix
        return(X + self._mean)
    
    def fit_transform(self, X):
        self.fit(X)
        return(self.transform(X))