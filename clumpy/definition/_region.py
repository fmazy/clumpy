#!/usr/bin/env python3
# -*- coding: utf-8 -*

# import numpy as np

from . import LandUseCoverLayer
from ..tools import path_split

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
                 verbose = 0):
        self.label = label
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
    
    def fit(self,
            luc_initial,
            luc_final,
            mask=None,
            distances_to_states = {}):
        """
        Fit the region.
        """
        for state, land in self.lands.items():
            land.fit(state = state,
                    luc_initial=luc_initial,
                    luc_final = luc_final,
                    mask = mask,
                    distances_to_states = distances_to_states)
    
    def transition_probabilities(self,
                                  transition_matrix,
                                  luc,
                                  mask=None,
                                  distances_to_states = {},
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
            
            if path_prefix is not None:
                path_prefix += '_'+str(state.value)
            
            ltp = land.transition_probabilities(state,
                                         luc = luc,
                                         mask = mask,
                                         P_v = P_v,
                                         palette_v = palette_v,
                                         distances_to_states=distances_to_states,
                                         path_prefix = path_prefix)
            
            if path_prefix is None:
                J[state] = ltp[0]
                P[state] = ltp[1]
                
        return(J, P)
    
    def allocation(self,
                   transition_matrix,
                   luc,
                   luc_origin = None,
                   mask=None,
                   distances_to_states={},
                   path=None):
        
        if luc_origin is None:
            luc_origin = luc
        
        if isinstance(luc_origin, LandUseCoverLayer):
            luc_origin_data = luc_origin.get_data()
        else:
            luc_origin_data = luc_origin
            
        if isinstance(luc, LandUseCoverLayer):
            luc_data = luc.get_data().copy()
        else:
            luc_data = luc
        
        for state, land in self.lands.items():
            P_v, palette_v = transition_matrix.get_P_v(state)
            
            land.allocation(state=state,
                           P_v=P_v,
                           palette_v=palette_v,
                           luc=luc_data,
                           luc_origin=luc_origin_data,
                           mask=mask,
                           distances_to_states=distances_to_states,
                           path=None)
            
        if path is not None:
            folder_path, file_name, file_ext = path_split(path)
            return(LandUseCoverLayer(label = 'file_name',
                                     data = luc_data,
                                     copy_geo = luc_origin,
                                     path = path,
                                     palette = luc_origin.palette))
        
        