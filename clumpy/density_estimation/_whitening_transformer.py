# -*- coding: utf-8 -*-

import numpy as np

class _WhiteningTransformer():
    """
    Whitening transformation in order to have a covariance matrix equal to the identity matrix.
    """
    def fit(self, X):
        """
        Fit the transformer

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data samples.

        Returns
        -------
        self : _WhiteningTransformer
            The fitted object.

        """
        self._mean = X.mean(axis=0)
        
        self._num_obs = X.shape[0]
        
        _, self._s, Vt = np.linalg.svd(X - self._mean, full_matrices=False)
        self._V = Vt.T
        
        self._transform_matrix = self._V @ np.diag(1 / self._s) * np.sqrt(self._num_obs-1)
        self._inverse_transform_matrix = np.diag(self._s)  @ self._V.T / np.sqrt(self._num_obs-1)
        
        self._transform_det = np.abs(np.linalg.det(self._transform_matrix))
        self._inverse_transform_det = np.abs(np.linalg.det(self._inverse_transform_matrix))
        
        # for compatibility with other preprocessing methods
        self.scale_ = self._inverse_transform_det
        
        return(self)
        
    def transform(self, X):
        """
        Transform.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data samples.

        Returns
        -------
        X_wt : ndarray of shape (n_samples, n_features)
            Transformed samples.

        """
        X = X - self._mean
        return(X @ self._transform_matrix)
    
    def inverse_transform(self, X):
        """
        inverse transform.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data samples in the Whitening Transform space.

        Returns
        -------
        X_inv_wt : ndarray of shape (n_samples, n_features)
            Inverse transformed samples.

        """
        X = X @ self._inverse_transform_matrix
        return(X + self._mean)
    
    def fit_transform(self, X):
        """
        fit and transform.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data samples.

        Returns
        -------
        X_wt : ndarray of shape (n_samples, n_features)
            Transformed samples.
        """
        self.fit(X)
        return(self.transform(X))