#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 13:51:30 2021

@author: frem
"""

import numpy as np
from ._generalized_allocation import generalized_allocation_rejection_test
from ._patcher import _weighted_neighbors
from ..definition import LandUseCoverLayer
from ..calibration._calibrator import _compute_P_v__u_y, _transition_probabilities

def generic_allocator(calibrators,
                      tms,
                      path):
    
    start_luc_layer = calibrators[0].start_luc_layer
    allocated_map = start_luc_layer.get_data().copy()
    
    for id_calibrator, calibrator in enumerate(calibrators):
        tm = tms[id_calibrator]
        
        # first estimation of P_v__u_y
        P_v__u_y, list_v__u = _compute_P_v__u_y(tm, calibrator, patches_adjustment=True)
        
        
        _generic_allocator_region_process(calibrator,
                                          tm,
                                          allocated_map,
                                          P_v__u_y = P_v__u_y,
                                          list_v__u = list_v__u)
            
            
    map_f = LandUseCoverLayer(name="",
                              path = path, 
                              copy_geo=start_luc_layer,
                              data = allocated_map)
    
    return(map_f)

def _generic_allocator_region_process(calibrator,
                                      tm,
                                      allocated_map,
                                      P_v__u_y,
                                      P_y__u_v,
                                      list_v__u,
                                      n_patches_tries = 10**3):
    
    list_v__u = list_v__u.copy()
    
    region_start_map = allocated_map.copy()
        
    for u in P_v__u_y.keys():
        # GART
        P = P_v__u_y[u].copy()
        
        
        J_Y__u = calibrator._J_Y_u[u].copy()
        Y__u = calibrator._Y_u[u].copy()
        
        id_J_to_keep = np.arange(J_Y__u.size)
        
        N_u = calibrator._J_Y_u[u].size
        N_v__u = {v: tm.get_value(u, v) * N_u for v in tm.list_v}
        
        print('target : ', [N_v__u[v] for v in list_v__u[u]])       
        
        # print(np.unique(V, return_counts=True))
        
        loop = True
        while loop:
            print('loop !')
            # the probabilities of no-transition are stacked to P
            # to allow no-transition through the GART
            print(P.shape, len(list_v__u[u]))
            V = generalized_allocation_rejection_test(np.hstack((P, 1 - P.sum(axis=1)[:, None])),
                                                      list_v__u[u] + [u])
            
            J = J_Y__u[V != u]
            V = V[V != u]
            
            
            
            # patch parameters
            areas = np.zeros(J.size)
            eccentricity_mean = np.zeros(J.size)
            eccentricity_std = np.zeros(J.size)
            
            for v in list_v__u[u]:
                j = V == v
                
                n_try = 0
                area_sum = 0
                
                best_id_patches_parameters = None
                min_relative_error = np.inf
                
                N_j  = j.sum()
                
                while n_try <= n_patches_tries:
                    id_patches_parameters = np.random.choice(calibrator._patches[u][v]['area'].size,
                                                            size = N_j,
                                                            replace=True)
                    
                    area_sum = calibrator._patches[u][v]['area'][id_patches_parameters].sum()
                    relative_error = np.abs(N_v__u[v] - area_sum) / N_v__u[v]
                    
                    if relative_error < min_relative_error:
                        min_relative_error = relative_error
                        best_id_patches_parameters = id_patches_parameters
                    
                    n_try += 1
                
                # print(v, N_j, n_try, calibrator._patches[u][v]['area'][best_id_patches_parameters].sum(), N_v__u[v], min_relative_error)
                
                areas[j] = calibrator._patches[u][v]['area'][best_id_patches_parameters]
                
                eccentricity_mean[j] = np.mean(calibrator._patches[u][v]['eccentricity'][best_id_patches_parameters])
                eccentricity_std[j] = np.std(calibrator._patches[u][v]['eccentricity'][best_id_patches_parameters])
            
            map_P_v__u_y = {}
            for id_vi, v in enumerate(list_v__u[u]):
                map_P_v__u_y[v] = np.zeros(allocated_map.shape)
                map_P_v__u_y[v].flat[calibrator._J_Y_u[u]] = P_v__u_y[u][:, id_vi]
            
            S = 0
            
            n_g = {v:0 for v in list_v__u[u]}
            
            j_to_exclude = []
            
            for id_j in np.random.choice(J.size, J.size, replace=False):
                
                v = V[id_j]
                s, j_allocated = _weighted_neighbors(map_i_data = region_start_map,
                                                    map_f_data = allocated_map,
                                                    map_P_vf__vi_z = map_P_v__u_y[V[id_j]],
                                                    j_kernel = J[id_j],
                                                    vi = u,
                                                    vf = v,
                                                    patch_S = areas[id_j],
                                                    eccentricity_mean = eccentricity_mean[id_j],
                                                    eccentricity_std = eccentricity_std[id_j],
                                                    neighbors_structure = 'rook',
                                                    avoid_aggregation = True,
                                                    nb_of_neighbors_to_fill = 3,
                                                    proceed_even_if_no_probability = True)
                
                j_to_exclude += j_allocated
                
                S += s
                
                if s == 0:
                    n_g[v] += 1
                
                N_v__u[v] -= s
                
            for v in list_v__u[u]:
                if n_g[v] == 0:
                    N_v__u[v] = 0
            
            
            if np.all(np.array(list(N_v__u.values())) == 0):
                # the process is over
                loop = False
            
            else:
                print('not achieved : ', [N_v__u[v] for v in list_v__u[u]])
                # the probabilities are adjusted
                tm__u = tm.select_u(u)
                
                closure = 1
                
                for v in list_v__u[u]:
                    tm__u.set_value(N_v__u[v] / N_u, u, v)
                    closure -= N_v__u[v] / N_u
                
                tm__u.set_value(closure, u, u)
            
                # the datas are selected
                id_J_to_keep = ~np.isin(J_Y__u, j_to_exclude)
                
                J_Y__u = J_Y__u[id_J_to_keep]
                Y__u = Y__u[id_J_to_keep, :]
                
                P_y__u = calibrator._estimate_P_y__u(Y_u = {u : Y__u},
                                                     only_list_u = [u])
                
                tm__u = tm__u.patches(calibrator._patches)
                
                P_v__u_y, list_v__u = _transition_probabilities(tm__u,
                                                                P_y__u, 
                                                                {u: P_y__u_v[u][id_J_to_keep, :]},
                                                                list_v__u,
                                                                verbose=1)
                
                P = P_v__u_y[u]
                
                
                
                        
        print([N_v__u[v] for v in list_v__u[u]])       

    return(allocated_map)