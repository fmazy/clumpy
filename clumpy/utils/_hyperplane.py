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
    def __init__(self,
                 w=None,
                 b=None,
                 positive_side_scalar=1):
        self.w = w
        self.b = b
        self.positive_side_scalar = positive_side_scalar
    
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

    def set_positive_side(self, P):
        norm_P = np.dot(P[None, :], self.w) + self.b

        if norm_P == 0:
            raise(ValueError("P should not belongs to the hyperplane."))

        if norm_P > 0:
            self.positive_side_scalar = 1
        else:
            self.positive_side_scalar = -1

    def side(self, X):
        norm_vec = np.dot(X, self.w) + self.b
        norm_vec *= self.positive_side_scalar

        return(norm_vec>0)
