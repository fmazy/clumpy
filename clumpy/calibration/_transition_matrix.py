#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:56:26 2021

@author: frem
"""

import numpy as np

def transition_matrix(v_u, list_v):
    list_u = list(v_u.keys())
    
    M = np.zeros((len(list_u), len(list_v)))
    
    for u, v in v_u.items():
        unique_v, count_v = np.unique(v, return_counts=True)
        index_v = [list_v.index(vi) for vi in unique_v]
        
        M[list_u.index(u), index_v] = count_v / v.size
    
    return(M)
    
    