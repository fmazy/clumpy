#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 09:52:53 2020

@author: frem
"""

import pandas as pd
import numpy as np
from scipy import ndimage

from . import explanatory_variable
from . import transition

def create_J(layer_LUC_i, layer_LUC_f=None, T=None):
    # self.layer_LUC_i = layer_LUC_i
    # self.T = T
    
    # all pixels with vi value
    cols = [('v', 'i')]
    cols = pd.MultiIndex.from_tuples(cols)
    
    J = pd.DataFrame(layer_LUC_i.data.flat, columns=cols)
    
    # restrict to pixels with vi in Ti
    if type(T) == transition.Transition:
        J = J.loc[J.v.i.isin(T.Ti.keys())]
    
    # complete with vf values
    if layer_LUC_f != None:
        add_vf(J, layer_LUC_f, inplace=True)
        
    return(J)

def add_vf(J,layer_LUC_f, inplace=False):
    if not inplace:
        J = J.copy()
    
    J['v', 'f'] = layer_LUC_f.data.flat[J.index.values]
     
    return(J)

def compute_N_vi_vf(J):
    return(J.groupby(['vi', 'vf']).size().reset_index(name='N_vi_vf'))

def compute_N_vi(J, nb_v=None):
    if nb_v == None:
        return(J.groupby([('v','i')]).size().reset_index(name='N_vi'))
    else:
        N_vi = J.groupby(['vi']).size().reset_index(name='N_vi')
        
        N_vi_full = pd.DataFrame(np.arange(nb_v), columns=['vi'])
        N_vi_full = N_vi_full.merge(right=N_vi,
                                    how='left',
                                    on=['vi'])
        N_vi_full.fillna(0, inplace=True)
        return(N_vi_full)

def restrict_vf_to_T(J, T, inplace=False):
    if not inplace:
        J_restricted = J.copy()
        
    N_vi_vf = J.groupby([('v','i'), ('v','f')]).size().reset_index(name=('N_vi_vf',''))
    # print(N_vi_vf)
    for index, row in N_vi_vf.iterrows():
        vi = row[('v','i')]
        vf = row[('v','f')]
        
        
        if vf not in T.Ti[vi].Tif.keys():
            if inplace:
                J.loc[(J.v.i==vi) & (J.v.f==vf), ('v','f')] = vi
            else:
                J_restricted.loc[(J_restricted.v.i==vi) & (J_restricted.v.f==vf), 'vf'] = vi
    
    
    
    # for Ti in T.Ti.values():
    #     if inplace:
    #         J.loc[J.vi == Ti.vi & (~J.vf.isin(Ti.Tif.keys())), 'vf'] = Ti.vi
    #     else:
    #         print(J_restricted.loc[J_restricted.vi == Ti.vi & (~J_restricted.vf.isin(Ti.Tif.keys())), 'vf'])
    #         J_restricted.loc[J_restricted.vi == Ti.vi & (~J_restricted.vf.isin(Ti.Tif.keys())), 'vf'] = Ti.vi
    
    if inplace:
        return(None)
    else:
        return(J_restricted)

def add_Zk_from_LayerEV(J, T, list_vi, layer_EV, name=None):
    """
    add an explanatory variable from a layer
    
    :param list_Tif: list of Tif
    :type list_Tif: [Transition_vi_vf]
    :param layer_EV: explanatory variable layer object
    :type layer_EV: LayerEV
    :param name: name -- default: None, ie takes the name of the EV layer
    :type name: string or None
    """
    
    if name == None:
        name = layer_EV.name
        
    add_Zk_from_numpy(J, T, list_vi, layer_EV.data, name, 'static')
    
def add_Zk_as_distance_to_v(J, T, list_vi, v, luc=None, luc_data=None, name=None, scale=1):
    """
    add an explanatory variable as a distance to a state
    
    :param list_Tif: list of Tif
    :type list_Tif: [Transition_vi_vf]
    :param v: state to compute distance
    :type v: int
    :param LUC_data: LUC data
    :type LUC_data: numpy array
    :param name: name
    :type name: string
    :param scale: size of a pixel in meter. not needed. default=1
    :type scale: float
    """
    
    if type(luc)==type(None) and type(luc_data)==type(None):
        print('error luc entry')
        return(False)
    
    if type(luc) != type(None) and type(luc_data) == type(None):
        luc_data = luc.data
    
    v_matrix = (luc_data == v).astype(int)
    distance = ndimage.distance_transform_edt(1 - v_matrix) * scale
    
    if name==None:
        name = 'distance_to_'+str(v)
    
    add_Zk_from_numpy(J, T, list_vi, distance, name, 'distance_to_v')
    
    
def add_Zk_from_numpy(J, T, list_vi, data, name, kind):
    
    # get all vi
    list_vi.sort()
    vi_T = list(T.Ti.keys())
    vi_T.sort()
        
    # if vi states asked represent all vi transitions, takes all indexes
    if list_vi == vi_T:
        J['z', name] = data.flat[J.index.values]
        
    else: # else, just add necessary                
        # then, for each vi
        for vi in list_vi:
            j = J.loc[J.v.i == vi].index.values
            J.loc[j, ('z', name)] = data.flat[j]
    
    J.sort_index(axis=1, inplace=True)
    
    # finally, we create for each vi the corresponding Z                
    for vi in list_vi:
        T.Ti[vi].Z[name] = explanatory_variable.Zk(name = name,
                                                        kind = kind,
                                                        Ti = T.Ti[vi])