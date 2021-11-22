#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 09:39:01 2021

@author: frem
"""

import numpy as np

class Hyperplane():
    """
    Hyperplane object. Used for boundary bias correction within the GKDE method.
    """
    
    def set_by_points(self, A):
        """
        Set the hyperplane by points.

        Parameters
        ----------
        A : array-like of shape (n_samples, n_features)
            The points which define the hyperplane. Each row is a point.
            For `$d$` dimension, ``A`` should be of shape `$(d,d)$`.

        Returns
        -------
        self : Hyperplane
            The set hyperplane.

        """
        if A.shape[0] != A.shape[1]:
            raise(ValueError('Unexpected A value. For a d dimension problem, it should be of shape (d,d).'))
        
        self.A = A
        self.w = np.linalg.solve(A, np.ones(A.shape[0]))
        self.b = - np.dot(self.w, A[0])
        
        return(self)
    
    def distance(self, X, p=2):
        """
        Computes the distance to the hyperplane according to the formula :
        
        ..math::
            dist = \frac{x \dot w + b}{\| w \|}

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The points to compute the distance to the hyperplane.

        Returns
        -------
        dist : array-like of shape (n_samples,).
            The computed distances.
        

        """
        dist = np.abs(np.dot(X, self.w) + self.b) / np.linalg.norm(self.w, ord=p)
        
        return(dist)
