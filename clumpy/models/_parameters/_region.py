#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class RegionParams():
    """
    Region parameters.

    Parameters
    ----------
    transition_matrix : TransitionMatrix
        The transition matrix for this region.
    """
    def __init__(self, transition_matrix):
        self.transition_matrix = transition_matrix
        self.lands_params = {}
        
    def __repr__(self):
        return('region_params:'+str(list(self.transitions)))
    
    def add_land_parameters(self, state, land_params):
        """
        Set all transitions for each initial states

        Parameters
        ----------
        state : State
            The final state.
        
        land : parameters.Land
            The land parameters.

        Returns
        -------
        self : RegionParams
            The self object.
        """
        self.lands_params[state] = land_params
        
        return(self)