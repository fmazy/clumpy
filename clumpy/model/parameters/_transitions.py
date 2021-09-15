#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ...definition import FeatureLayer, Palette
from ...density_estimation import Parameters as DensityEstimationParameters

class Transitions():
    """
    Transitions parameters used by module classes.

    Parameters
    ----------
    features : list of FeaturesLayer or State
        List of features where a State means a distance layer to the corresponding state.
    dep_P_x__u : density_estimation.Parameters, default=None
        Density estimation parameters (dep) for :math:`P(x|u)`. If None, default
        density estimation parameters are set.
    """
    def __init__(self, 
                 features,
                 dep_P_x__u = None):
        
        self.features = features
        
        self.dep_P_x__u = dep_P_x__u
        if self.dep_P_x__u is None:
            self.dep_P_x__u = DensityEstimationParameters()
        
        self.dep_P_x__u_v = {}
        
    def __repr__(self):
        return('transitions->'+str(list(self.dep_P_x__u_v.keys())))
             
    def add_density_estimation_parameters(self, state, dep=None):
        """
        Set density estimation parameters for each transitions

        Parameters
        ----------
        state : State
            The final state.
        dep : density_estimation.Parameters
            Density estimation parameters. If None, default density estimation parameters
            is set.

        Returns
        -------
        self : Transitions
            The self object.
        """
        self.dep_P_x__u_v[state] = dep
    

    
    