#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 16:54:18 2021

@author: frem
"""


def transition_probabilities(luc_initial,
                            luc_final,
                            luc_start,
                            regions,
                            transition_matrices,
                            folder_path):
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
    regions : TYPE
        DESCRIPTION.
    transition_matrices : TYPE
        DESCRIPTION.
    path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    distances_to_states_calibration = {}
    distances_to_states_allocation = {}
    
    for id_region, region in enumerate(regions):
        region.transition_probabilities(luc_initial=luc_initial,
                                          luc_final=luc_final,
                                          luc_start=luc_start,
                                          transition_matrix=transition_matrices[id_region],
                                          distances_to_states_calibration = distances_to_states_calibration,
                                          distances_to_states_allocation = distances_to_states_allocation,
                                          path_prefix = folder_path+region.label)
    