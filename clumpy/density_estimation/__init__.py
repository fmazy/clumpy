"""Density Estimation Methods"""

import numpy as np

from ekde import KDE

_methods = {'kde':KDE,}

class NullEstimator():
    def __init__(self):
        super().__init__()
        
    def __repr__(self):
        return("NullEstimator()")
    
    def fit(self, X, y=None):
        return (self)

    def predict(self, X):
        return (np.zeros(X.shape[0]))
    
    def set_params(self, **params):
        """
        Set parameters.

        Parameters
        ----------
        **params : kwargs
            Parameters et values to set.

        Returns
        -------
        self : DensityEstimator
            The self object.

        """
        for param, value in params.items():
            setattr(self, param, value)