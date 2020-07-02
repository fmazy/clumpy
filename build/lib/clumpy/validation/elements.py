#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 09:58:28 2020

@author: frem
"""

import numpy as np

def create_similarity_matrix(v_dim):
    M = np.zeros((v_dim+1,v_dim+1,v_dim+1,v_dim+1))
    for vi in range(v_dim+1):
        for vf in range(v_dim+1):
            M[vi,vf,vi,vf] = 1
    
    return(M)

def add_similarity(M, x, vi1, vf1, vi2, vf2, inplace=True):
    if inplace==False:
        M = M.copy()
    
    M[vi1, vf1, vi2, vf2] = x
    M[vi2, vf2, vi1, vf1] = x
    
    if inplace==False:
        return(M)