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

def generic_allocator(calibrators, path):
    
    start_luc_layer = calibrators[0].start_luc_layer
    luc_data = start_luc_layer.get_data().copy()
    
    for calibrator in calibrators:
        
        sub_map = luc_data.copy()
        
        for u in calibrator._P_v__u_y.keys():
            
            
            
            # GART
            P = calibrator._P_v__u_y[u]
            P = np.hstack((P, 1 - P.sum(axis=1)[:, None]))
            
            list_v__u = calibrator._list_v__u[u] + [u]
            
            V = generalized_allocation_rejection_test(P, list_v__u)
            
            print(np.unique(V, return_counts=True))
            
            J = calibrator._J_Y_u[u][V != u]
            V = V[V != u]
            
            # patch parameters
            areas = np.zeros(J.size)
            eccentricity_mean = np.zeros(J.size)
            eccentricity_std = np.zeros(J.size)
            
            for v in calibrator._list_v__u[u]:
                j = V == v
                
                id_patches_parameters = np.random.choice(calibrator._patches[u][v]['area'].size,
                                                        size = j.sum(),
                                                        replace=True)
                
                areas[j] = calibrator._patches[u][v]['area'][id_patches_parameters]
                eccentricity_mean[j] = np.mean(calibrator._patches[u][v]['eccentricity'][id_patches_parameters])
                eccentricity_std[j] = np.std(calibrator._patches[u][v]['eccentricity'][id_patches_parameters])
            
            map_P_v__u_y = {}
            for id_vi, v in enumerate(calibrator._list_v__u[u]):
                map_P_v__u_y[v] = np.zeros(luc_data.shape)
                map_P_v__u_y[v].flat[calibrator._J_Y_u[u]] = calibrator._P_v__u_y[u][:, id_vi]
            
            S = 0
            cnt = 0
            for id_j in np.random.choice(J.size, J.size, replace=False):
                
                
                
                s = _weighted_neighbors(map_i_data = sub_map,
                                        map_f_data = luc_data,
                                        map_P_vf__vi_z = map_P_v__u_y[V[id_j]],
                                        j_kernel = J[id_j],
                                        vi = u,
                                        vf = V[id_j],
                                        patch_S = areas[id_j],
                                        eccentricity_mean = eccentricity_mean[id_j],
                                        eccentricity_std = eccentricity_std[id_j],
                                        neighbors_structure = 'rook',
                                        avoid_aggregation = True,
                                        nb_of_neighbors_to_fill = 3,
                                        proceed_even_if_no_probability = True)
                
                if s == 0:
                    print(cnt, '!', J[id_j], u, V[id_j], areas[id_j])
                cnt += 1
                S += s
            
            
    map_f = LandUseCoverLayer(name="",
                              path = path, 
                              copy_geo=start_luc_layer,
                              data = luc_data)
    
    return(map_f)
            # print('number of pivot cells : ', (v != u).sum())
            
            # areas = calibrator._patches[u]
            
            # calibrator._J_Y_u.size
            # print(u)