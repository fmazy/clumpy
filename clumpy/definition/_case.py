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
from copy import deepcopy
from sys import getsizeof

from ._transition import _Transition
from ._feature import _Zk
from ..tools import human_size
from ..tools import np_suitable_integer_type

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
        
        self.layers = ['vi']
        
        self.J = {}
        if map_f is not None:
            self.vf = {}
        else:
            self.vf = None
        self.Z = {}
        self.Z_names = {}
        
        for vi in dict_vi_vf.keys():
            self.J[vi] = np_suitable_integer_type(np.where(self.map_i.data.flat==vi)[0])
            
            if map_f is not None:
                self.vf[vi] = np_suitable_integer_type(self.map_f.data.flat[self.J[vi]])
                
                if restrict_vf_to_studied_ones:
                    self.vf[vi][~np.isin(self.vf[vi], dict_vi_vf[vi])] = vi
                    
            self.Z[vi] = None
            self.Z_names[vi] = None

    
    def copy(self):
        return(deepcopy(self))
    
    def get_size(self, print_value=True, human=True, return_value=False):
        s = 0
        for vi in self.J.keys():
            s += getsizeof(self.J[vi])
            s += getsizeof(self.vf[vi])
            s += getsizeof(self.Z[vi])
            
        if print_value:
            if human:
                sh = human_size(s)
                print(str(round(sh[0],2))+' '+sh[1])
            else:
                print(str(s)+' B')
        
        if return_value:
            return(s)
               
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
        for vi in list_vi:
            if type(self.Z[vi]) == type(None):
                self.Z[vi] = data.flat[self.J[vi]]
                self.Z_names[vi] = [name]
            else:
                self.Z[vi] = np.column_stack((self.Z[vi],
                                              data.flat[self.J[vi]]))
                self.Z_names[vi].append(name)
        

            
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
            
    def get_z_column_id(self, vi, z_name):
        return(self.Z_names[vi].index(z_name))
    
    def get_z(self, vi, z_name):
        
        column_id = self.get_z_column_id(vi, z_name)
        
        return(self.Z[vi][:,column_id])
    
    def get_J(self, vi):
        return(self.J[vi])
    
    def get_vf(self, vi):
        return(self.vf[vi])
    
    def remove(self, vi, condition, inplace=False):
        return(self.keep_only(vi, ~condition, inplace=inplace))
        
    def keep_only(self, vi, condition, inplace=False):
        if inplace:
            case = self
        else:
            case = self.copy()
        
        case.J[vi] = case.J[vi][condition]
        case.Z[vi] = case.Z[vi][condition,:]
        
        if case.vf is not None:
            case.vf[vi] = case.vf[vi][condition]
        
        if not inplace:
            return(case)
        
    def select_vi(self, vi, inplace=False):
        if inplace:
            case = self
        else:
            case = self.copy()
        
        vix_list = list(case.J.keys())
        for vix in vix_list:
            if vix != vi:
                case.J.pop(vix)
                case.Z_names.pop(vix)
                case.Z.pop(vix)
                case.dict_vi_vf.pop(vix)
                
                if case.vf is not None:
                    case.vf.pop(vix)
            
        if not inplace:
            return(case)
        
    def remove_z(self, vi, z_name, inplace=False):
        if inplace:
            case = self
        else:
            case = self.copy()
        
        column_id = self.get_z_column_id(vi, z_name)
        
        list_columns = list(range(case.Z[vi].shape[1]))
        list_columns.pop(column_id)
        
        case.Z[vi] = case.Z[vi][:, list_columns]
        
        case.Z_names[vi].pop(column_id)
        
        if not inplace:
            return(case)
    
    def get_z_as_dataframe(self):
        z = {}
        for vi in self.Z.keys():
            col_names = pd.MultiIndex.from_tuples([('z',z_name) for z_name in self.Z_names[vi]])
            z[vi] = pd.DataFrame(self.Z[vi], columns=col_names)
        
        return(z)
    
    def get_unique_z(self, output='np'):
        z = self.get_z_as_dataframe()
        
        for vi in z.keys():
            z[vi].drop_duplicates(inplace=True)
            z[vi].reset_index(drop=True, inplace=True)
            
            if output=='np':
                z[vi] = z[vi].values
        
        return(z)

def get_pixels_coordinates(J, map_shape):
    x, y = np.unravel_index(J.index.values, map_shape)
    coor = np.zeros((x.size,2))
    coor[:,0] = x
    coor[:,1]  = y
    return(coor)
            