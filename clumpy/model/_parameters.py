#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 17:23:17 2021

@author: frem
"""

class Parameters():
    def __init__(self):
        self.regions = []
        self.transition_matrixes = {}
        self.list_v__u = {}
        self.features__u = {}
        
    def add_region(self,
                   region):
        
        self.regions.append(region)
        
        return(self)
    
    def set_transition_matrix(self,
                              tm,
                              region):
        
        self.transition_matrixes[region] = tm
        
        self.list_v__u[region] = {}
        self.features__u[region] = {}
        
        for u in tm.list_u:
            self.list_v__u[region][u] = []
            for v in tm.list_v:
                if tm.get_value(u, v) > 0:
                    self.list_v__u[region][u].append(v)
        
        return(self)
    
    def add_layer_feature(self,
                        region,
                        u,
                        feature_layer):
        
        if u not in self.features__u[region].keys():
            self.features__u[region][u] = []
    
        self.features__u[region][u].append(('layer', feature_layer))
        
    def add_distance_feature(self,
                            region,
                            u,
                            e):
        
        if u not in self.features__u[region].keys():
            self.features__u[region][u] = []
    
        self.features__u[region][u].append(('distance', e))
        
        
    
    def get_case_params(self, region):
        params = {}
        for u, list_v__u in self.list_v__u[region].items():
            params[u] = {'v' : list_v__u,
                         'features' : self.features__u[region][u]}
            
        return(params)
    
class RegionParameters():
    def __init__(self):
        self.state_parameters = {}
    
    def add_transition_matrix(self, tm):
        self.transition_matrix = tm
    
    def add_state_parameters(self, u, state_parameters):
        self.state_parameters[u] = state_parameters

class StateParameters():
    def __init__(self):
        self.features = []
    
    def add_layer_feature(self, feature_layer):
        self.features.append(('layer', feature_layer))
        
    def add_distance_feature(self, e):
        self.features.append(('distance', e))

    
    