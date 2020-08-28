#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 09:35:47 2020

@author: frem
"""

# import numpy as np
import pandas as pd
import numpy as np
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
        
    dict_vi_vf : list of tuples
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
    def __init__(self, map_i, dict_vi_vf, map_f=None, restrict_vf_to_studied_ones = True):
        self.map_i = map_i
        self.map_f = map_f
        
        self.dict_vi_vf = dict_vi_vf
        
        self._create_J()
        
        if restrict_vf_to_studied_ones == True and type(map_f) != type(None):
            self._restrict_vf_to_studied_ones()
        
        # self.transitions = _Transition()
        # for vi_vf in dict_vi_vf:
        #     self.transitions.addTif(vi=vi_vf[0], vf=vi_vf[1])

        # self._create_J()
        
        # if restrict_vf_to_studied_ones and type(map_f) != type(None):
        #     self._restrict_vf_to_studied_ones()
            
    def _create_J(self):
        # self.layer_LUC_i = layer_LUC_i
        # self.T = T
        
        # all pixels with vi value
        cols = [('v', 'i')]
        cols = pd.MultiIndex.from_tuples(cols)
        
        self._J = pd.DataFrame(columns=cols)
        
        for vi in self.dict_vi_vf.keys():
            J_vi = pd.DataFrame(columns=cols)
            
            J_vi[('j','')] = np.where(self.map_i.data.flat==vi)[0]
            J_vi[('v','i')] = vi
            J_vi.set_index('j', inplace=True)
            
            self._J = pd.concat([self._J, J_vi])
        
        if type(self.map_f) != type(None):
            self._J[('v','f')] = self.map_f.data.flat[self._J.index.values]
        
        
    def get_J(self, copy=True):
        if copy:
            return(self._J.copy())
        else:
            return(self._J)
            
    def _restrict_vf_to_studied_ones(self):
        for vi in self.dict_vi_vf.keys():
            self._J.loc[(self._J.v.i == vi) & ~(self._J.v.f.isin(self.dict_vi_vf[vi])), ('v','f')] = vi    
    
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
        vi_T = list(self.dict_vi_vf.keys())
        vi_T.sort()
            
        # if vi states asked represent all vi transitions, takes all indexes
        if list_vi == vi_T:
            self._J['z', name] = data.flat[self._J.index.values]
            
        else: # else, just add necessary                
            # then, for each vi
            for vi in list_vi:
                j = self._J.loc[self._J.v.i == vi].index.values
                self._J.loc[j, ('z', name)] = data.flat[j]
        
        self._J.sort_index(axis=1, inplace=True)
        
        # # finally, we create for each vi the corresponding Z                
        # for vi in list_vi:
        #     self.transitions.Ti[vi].Z[name] = _Zk(name = name,
        #                                                     kind = kind,
        #                                                     Ti = self.transitions.Ti[vi])
            
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
        
    # def standard_scale(self):
    #     """
    #     Standardize features by removing the mean and scaling to unit variance for each initial states.
        
    #     Notes
    #     -----
    #         New attributes are then available :
        
    #             ``self._J_standard``
    #                 The standardized features.
    #     """
    #     scaler = SklearnStandardScaler()  # doctest: +SKIP
    #     # Don't cheat - fit only on training data
    #     scaler.fit(X_train)  # doctest: +SKIP
    #     X_train = scaler.transform(X_train)  # doctest: +SKIP
    #     # apply same transformation to test data
    #     X_test = scaler.transform(X_test)  # doctest: +SKIP
        
        