#!/usr/bin/env python3
# -*- coding: utf-8 -*

import numpy as np

class Region():
    """
    Define a region.

    Parameters
    ----------
    label : str
        The region's label. It should be unique.
        
    luc_initial : LandUseLayer
        The initial land use where the calibration is made.
    
    luc_final : LandUseLayer, default=None
        The final land use where the calibration is made.
    
    luc_start : LandUseLayer, default=None
        The starting land use where the calibration is made.
        
    region_calibration : LandUseLayer, default=None
        The region mask layer used for calibration.
        
    region_allocation : LandUseLayer, default=None
        The region mask used for allocation.
    
    verbose : int, default=0
        Verbosity level.
    """
    def __init__(self,
                 label,
                 transition_matrix=None,
                 mask_calibration = None,
                 mask_allocation = None,
                 verbose = 0):
        self.label = label
        self.mask_calibration = mask_calibration
        self.mask_allocation = mask_allocation
        self.verbose = verbose
        
        self.lands = {}
        
    def __repr__(self):
        return(self.label)
    
    def add_land(self, state, land):
        """
        Add a land for a given state.

        Parameters
        ----------
        state : State
            The initial state.
        
        land : Land
            The Land object.

        Returns
        -------
        self : Region
            The self object.

        """
        self.lands[state] = land
        
        return(self)
    
    def transition_probabilities(self,
                                  luc_initial,
                                  luc_final,
                                  luc_start,
                                  transition_matrix,
                                  distances_to_states_calibration = {},
                                  distances_to_states_allocation = {},
                                  path_prefix=None):
        """
        Compute transition probabilities.

        Parameters
        ----------
        luc_initial : TYPE
            DESCRIPTION.
        luc_final : TYPE
            DESCRIPTION.
        luc_start : TYPE
            DESCRIPTION.
        mask_calibration : TYPE
            DESCRIPTION.
        mask_allocation : TYPE
            DESCRIPTION.
        tm : TYPE
            DESCRIPTION.
        out : TYPE, optional
            DESCRIPTION. The default is None.
        distances_to_states_calibration : TYPE, optional
            DESCRIPTION. The default is {}.
        distances_to_states_allocation : TYPE, optional
            DESCRIPTION. The default is {}.

        Returns
        -------
        None.

        """
        
        J = {}
        P = {}
        
        for state, land in self.lands.items():
            
            P_v, palette_v = transition_matrix.get_P_v(state)
            
            if path_prefix is None:
                path_prefix += '_'+str(state.value)
            
            ltp = land.transition_probabilities(state,
                                         luc_initial = luc_initial,
                                         luc_final = luc_final,
                                         luc_start = luc_start,
                                         mask_calibration = self.mask_calibration,
                                         mask_allocation = self.mask_allocation,
                                         P_v = P_v,
                                         palette_v = palette_v,
                                         distances_to_states_calibration=distances_to_states_calibration,
                                         distances_to_states_allocation=distances_to_states_allocation,
                                         path_prefix = path_prefix)
            
            if path_prefix is None:
                J[state] = ltp[0]
                P[state] = ltp[1]
                
        return(J, P)
    
        