#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 08:47:24 2021

@author: frem
"""

import os

from ..definition import LandUseCoverLayer
from ..tools import path_split

def allocation(luc_initial,
               luc_final,
               luc_start,
               regions,
               params_regions,
               path):
    """
    Allocation function according to regions parameters.

    Parameters
    ----------
    luc_initial : LandUseCoverLayer
        Initial land use for calibration.
    luc_final : LandUseCoverLayer
        Final land use for calibration.
    luc_start : LandUseCoverLayer
        Start land use for allocation.
    regions : list(Region)
        List of Region objects.
    params_regions : list(parameters.Region)
        List of region parameters objects.
    path : str
        The path to save the allocated map (tif file).
        
    Returns
    -------
    luc_allocated : LandUseCoverLayer
        Allocated land use.
    """
    
    # Path process
    folder_path, file_name, file_ext = path_split(path)
    
    allocated_map = luc_start.get_data().copy()
    
    for id_region, region in enumerate(regions):
        allocated_map = _region_allocation(luc_initial = luc_initial,
                                            luc_final = luc_final,
                                            region = region,
                                            params_region = params_regions[id_region],
                                            allocated_map = allocated_map)
    
    luc_allocated = LandUseCoverLayer(label=file_name,
                                      path = path, 
                                      copy_geo=luc_start,
                                      data = allocated_map)
    
    return(luc_allocated)
        
    
def _region_allocation(luc_initial,
                      luc_final,
                      region,
                      params_region,
                      allocated_map):
    """
    Allocation function dedicated to only one region.

    Parameters
    ----------
    luc_initial : LandUseCoverLayer
        Initial land use for calibration.
    luc_final : LandUseCoverLayer
        Final land use for calibration.
    regions : Region
        Region.
    params_regions : Region
        Region Parameters.
    allocated_map : array-like of type int
        The allocated map as an array-like data.
        
    Returns
    -------
    allocated_map : ndarray
        The allocated map data.
    """
    
    return(allocated_map)