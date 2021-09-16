#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ...definition import FeatureLayer, Palette
from ...density_estimation import DensityEstimationParams

class LandParams():
    """
    Land parameters used by module classes.

    Parameters
    ----------
    features : list(FeaturesLayer) or list(State)
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
            self.dep_P_x__u = DensityEstimationParams()
        
        self.dep_P_x__u_v = {}
        self.patches_params = {}
        
    def __repr__(self):
        return('trans_params->'+str(list(self.dep_P_x__u_v.keys())))
             
    def add_density_estimation_parameters(self, state, dep=None):
        """
        Set density estimation parameters for each final state.

        Parameters
        ----------
        state : State
            The final state.
        dep : DensityEstimationParams
            Density estimation parameters. If None, default density estimation parameters
            is set.

        Returns
        -------
        self : Land
            The self object.
        """
        self.dep_P_x__u_v[state] = dep
        
        return(self)
    
    def add_patches_params(self, state, patches_params):
        """
        Set density estimation parameters for each final state.

        Parameters
        ----------
        state : State
            The final state.
        patches_params : PatchesParams
            The patches parameters.

        Returns
        -------
        self : Land
            The self object.
        """
        self.patches_params[state] = patches_params
    
        return(self)
    
    