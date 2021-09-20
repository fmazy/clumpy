#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 15:40:56 2021

@author: frem
"""

from . import LandUseLayer
from ..tools import path_split

class Territory():
    def __init__(self,
                 regions = None):
        
        self.regions = regions
        if self.regions is None:
            self.regions = {}
    
    def add_regions(self, region):
        self.regions[region] = region
    
    def remove_regions(self, region):
        self.regions.remove(region)

    def check(self):
        """
        Check the Region object through regions checks.
        Notably, estimators uniqueness are checked to avoid malfunctioning during transition probabilities estimation.
        """
        density_estimators = []
        feature_selectors = []
        for region in self.regions:
            density_estimators = region._check_density_estimators(density_estimators=density_estimators)
            feature_selectors = region._check_feature_selectors(feature_selectors=feature_selectors)


    def fit(self,
            luc_initial,
            luc_final,
            masks=None):
        
        if masks is None:
            masks = {region:None for region in self.regions}
        
        distances_to_states = {}
        
        for id_region, region in enumerate(self.regions):
            region.fit(luc_initial = luc_initial,
                       luc_final = luc_final,
                       mask = masks[region],
                       distances_to_states = distances_to_states)
    
    def transition_probabilities(self,
                                 transition_matrices,
                                 luc,
                                 masks=None,
                                 path_prefix=None):
        
        if masks is None:
            masks = {region:None for region in self.regions}
        
        distances_to_states = {}
        
        tp = {}
        
        for region in self.regions:
            
            if path_prefix is not None:
                path_prefix += '_'+str(region.label)
            
            tp[region] = region.transition_probabilities(transition_matrix = transition_matrices[region],
                                            luc = luc,
                                            mask=masks[region],
                                            distances_to_states = distances_to_states,
                                            path_prefix=path_prefix)
            
        if path_prefix is None:
            return(tp)
        
    def allocation(self,
                   transition_matrices,
                   luc,
                   masks=None,
                   path=None):
        
        if masks is None:
            masks = {region:None for region in self.regions}
            
        distances_to_states = {}
        
        luc_data = luc.get_data().copy()
        
        for region in self.regions:
            region.allocation(transition_matrix = transition_matrices[region],
                              luc = luc_data,
                              luc_origin = luc,
                              mask=masks[region],
                              distances_to_states = distances_to_states,
                              path=path)
            
        if path is not None:
            folder_path, file_name, file_ext = path_split(path)
            return(LandUseLayer(label = 'file_name',
                                     data = luc_data,
                                     copy_geo = luc,
                                     path = path,
                                     palette = luc.palette))
        
        return(luc_data)
    
    def multisteps_allocation(self,
                              n,
                              transition_matrices,
                              luc,
                              masks=None,
                              path_prefix=None):
        
        multisteps_transition_matrices = {region : tm.multisteps(n) for region, tm in transition_matrices.items()}
        
        luc_step = luc
        
        for i in range(n):
            
            if isinstance(path_prefix, str):
                path_step = path_prefix + '_' + str(i) + '.tif'
            elif callable(path_prefix):
                path_step = path_prefix(i)
            
            luc_step = self.allocation(transition_matrices = multisteps_transition_matrices,
                            luc = luc_step,
                            masks = masks,
                            path = path_step)
        
        return(luc_step)
