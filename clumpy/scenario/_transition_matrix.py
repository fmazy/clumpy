#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 10:47:48 2021

@author: frem
"""

import numpy as np

def multisteps_transition_matrix(M, list_u, list_v, n):
    
    max_v = np.max(list_v)
        
    transition_matrix = np.diag(np.ones(max_v+1))
    
    list_v_full_matrix = list(np.arange(max_v+1))
    
    for id_u, u in enumerate(list_u):
        for id_v, v in enumerate(list_v):
            transition_matrix[list_v_full_matrix.index(u),
                              list_v_full_matrix.index(v)] = M[id_u, id_v]
            
            # transition_matrix[list_v_full_matrix.index(u),
            #                   list_v_full_matrix.index(u)] -= M[id_u, id_v]
    
    
    tp = transition_matrix.astype(float)
    
    tp = np.nan_to_num(tp)
    
    eigen_values, P = np.linalg.eig(tp)
    
    
    # print(eigen_values)
    eigen_values = np.power(eigen_values, 1/n)
    
    ms_tp = np.dot(np.dot(P, np.diag(eigen_values)), np.linalg.inv(P))
    
    M_multisteps = np.zeros_like(M)
    
    for id_u, u in enumerate(list_u):
        for id_v, v in enumerate(list_v):
            M_multisteps[id_u, id_v] = ms_tp[list_v_full_matrix.index(u),
                                             list_v_full_matrix.index(v)]
            
    return(M_multisteps)

def patches_transition_matrix(M, list_u, list_v, patches):
    M_p = np.zeros_like(M)
    M_p.fill(1.0)
    
    for id_u, u in enumerate(list_u):
        for id_v, v in enumerate(list_v):
            if u != v:
                M_p[id_u, id_v] = M[id_u, id_v] / patches[u][v]['area'].mean()
                M_p[id_u, list_v.index(u)] -= M_p[id_u, id_v]
    
    return(M_p)