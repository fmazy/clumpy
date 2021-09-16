#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from scipy import ndimage
import numpy as np

class Region():
    def __init__(self,
                 label,
                 luc_initial,
                 luc_final = None,
                 luc_start = None,
                 region_calibration = None,
                 region_allocation=None,
                 verbose = 0):
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
        self.label = label
        self.luc_initial = luc_initial
        self.luc_final = luc_final
        self.luc_start = luc_start
        self.region_calibration = region_calibration
        self.region_allocation = region_allocation
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
    
    
        