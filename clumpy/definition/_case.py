#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 09:35:47 2020

@author: frem
"""

import numpy as np
import pandas as pd
from scipy import ndimage

from ._transition import _Transition
from ._feature import _Zk

class Case():
    """
    A land use and cover change model case.
    
    Parameters
    ----------
    map_i : [LayerLUC]
        The initial LUC map.
        
    list_vi_vf : list of tuples
        The list of studied transitions represented as tuples. Example : ``[(3,2), (3,4), (7,2)]``
            
    map_f : [LayerLUC] or None (default=None)
        The final LUC map. Can be None.
        
    restrict_vf_to_studied_ones : Boolean
        If ``True``, pixels final states are restricted to the studied ones.
    
    Attributes
    ----------
    J : Pandas DataFrame
        The studied pixels through defined layers.
    
    transitions : [_Transition]
        The transitions definitions
    
    """
    def __init__(self, map_i, list_vi_vf, map_f=None, restrict_vf_to_studied_ones = True):
        self.map_i = map_i
        self.map_f = map_f
        
        self.transitions = _Transition()
        for vi_vf in list_vi_vf:
            self.transitions.addTif(vi=vi_vf[0], vf=vi_vf[1])

        self._create_J()
        
        if restrict_vf_to_studied_ones and type(map_f) != type(None):
            self._restrict_vf_to_studied_ones()
            
    def _create_J(self):
        # self.layer_LUC_i = layer_LUC_i
        # self.T = T
        
        # all pixels with vi value
        cols = [('v', 'i')]
        cols = pd.MultiIndex.from_tuples(cols)
        
        self.J = pd.DataFrame(self.map_i.data.flat, columns=cols)
        
        # restrict to pixels with vi in Ti
        self.J = self.J.loc[self.J.v.i.isin(self.transitions.Ti.keys())]
        
        # complete with vf values
        if self.map_f != None:
            self._add_vf()
            
    def _add_vf(self):       
        self.J['v', 'f'] = self.map_f.data.flat[self.J.index.values]
        
    def _restrict_vf_to_studied_ones(self):            
        N_vi_vf = self.J.groupby([('v','i'), ('v','f')]).size().reset_index(name=('N_vi_vf',''))
        # print(N_vi_vf)
        for index, row in N_vi_vf.iterrows():
            vi = row[('v','i')]
            vf = row[('v','f')]
            
            if vf not in self.transitions.Ti[vi].Tif.keys():
                self.J.loc[(self.J.v.i==vi) & (self.J.v.f==vf), ('v','f')] = vi       
            
    def add_distance_to_v_as_feature(self, list_vi, v, name=None, scale=1):
        """
        add an explanatory variable as a distance to a state
        
        Parameters
        ==========
        list_vi : list of int
            The list of concerned initial states for this feature. Example : ``[2,3]``.
        v : int
            The focused distance state.
        name : str (default=None)
            The feature name. If ``None``, the name will be ``'distance_to_'+str(v)``.
        scale : float (default=1)
            The pixel side length in meters.
        """
               
        v_matrix = (self.map_i.data == v).astype(int)
        distance = ndimage.distance_transform_edt(1 - v_matrix) * scale
        
        if name==None:
            name = 'distance_to_'+str(v)
        
        self.add_numpy_as_feature(list_vi, distance, name, 'distance_to_v')
        
    def add_numpy_as_feature(self, list_vi, data, name, kind='static'):
        """
        Parameters
        ----------
        list_vi : list of int
            The list of concerned initial states for this feature. Example : ``[2,3]``.
        data : numpy array
            The data whose shape is like the initial LUC map one's.
        name : str
            The feature name.
        kind : {'static', 'dyn', 'distance_to_v'}, (default='static')
            The kind of the feature.
            
            static
                Static feature.
            dyn
                Dynamic feature which should be recomputed at each allocation time step.
            distance_to_v
                The same as dyn but specific for distance to a state features.
        """
        # get all vi
        list_vi.sort()
        vi_T = list(self.transitions.Ti.keys())
        vi_T.sort()
            
        # if vi states asked represent all vi transitions, takes all indexes
        if list_vi == vi_T:
            self.J['z', name] = data.flat[self.J.index.values]
            
        else: # else, just add necessary                
            # then, for each vi
            for vi in list_vi:
                j = self.J.loc[self.J.v.i == vi].index.values
                self.J.loc[j, ('z', name)] = data.flat[j]
        
        self.J.sort_index(axis=1, inplace=True)
        
        # finally, we create for each vi the corresponding Z                
        for vi in list_vi:
            self.transitions.Ti[vi].Z[name] = _Zk(name = name,
                                                            kind = kind,
                                                            Ti = self.transitions.Ti[vi])
            
    def add_layer_as_feature(self, list_vi, layer_EV, name=None):
        """
        add an explanatory variable from a layer
        
        :param list_Tif: list of Tif
        :type list_Tif: [_Transition_vi_vf]
        :param layer_EV: explanatory variable layer object
        :type layer_EV: LayerEV
        :param name: name -- default: None, ie takes the name of the EV layer
        :type name: string or None
        """
        
        if name == None:
            name = layer_EV.name
            
        self.add_numpy_as_feature(list_vi, layer_EV.data, name)