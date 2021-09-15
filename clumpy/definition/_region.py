#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 17:45:24 2021

@author: frem
"""

class Region():
    def __init__(self,
                 label,
                 calibration_region,
                 allocation_region):
        """
        Define a region.

        Parameters
        ----------
        label : str
            The region's label. It should be unique.
        calibration_region : LandUseLayer
            The region where the calibration is made.
        allocation_region : LandUseLayer
            The region where the allocation is made.
        """
        self.label = label
        self.calibration_region = calibration_region
        self.allocation_region = allocation_region
        
    def __repr__(self):
        return(self.label)