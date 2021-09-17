#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 07:44:29 2021

@author: frem
"""

import numpy as np

from ._gart import generalized_allocation_rejection_test
from ._patcher import _weighted_neighbors
# def land_allocation(state,
#                     )

def try_land_allocation(state,
                        area_total,
                        J,
                        P_v__u_Y,
                        palette_v,
                        luc_ref,
                        luc_data,
                        transition_patches_isl_exp,
                        isl_exp,
                        n_patches_tries=1000):
         
    # GART
    # the probabilities of no-transition are stacked to P
    # to allow no-transition through the GART
    V = generalized_allocation_rejection_test(P_v__u_Y, palette_v.get_list_of_values())
    
    id_pivot = V != state.value
    V_pivot = V[id_pivot]
    J_pivot = J[id_pivot]
    
    palette_v_without_u = palette_v.remove(state, inplace=False)
    
    areas = {}
    eccentricities = {}
    for state_v in palette_v_without_u:
        areas[state_v], eccentricities[state_v] = draw_areas_eccentricities(area_total[state_v],
                                                                            transition_patches_isl_exp[state_v],
                                                                            J_pivot.size)
    
    
    
    for id_j in np.random.choice(J_pivot.size, J_pivot.size, replace=False):
        v = V_pivot[id_j]
        s, j_allocated = _weighted_neighbors(map_i_data = luc_ref,
                                            map_f_data = luc_data,
                                            map_P_vf__vi_z = map_P_v__Y[v],
                                            j_kernel = J_pivot[id_j],
                                            vi = u,
                                            vf = v,
                                            patch_S = areas[id_j],
                                            eccentricity_mean = eccentricity_mean[v],
                                            eccentricity_std = eccentricity_std[v],
                                            neighbors_structure = neighbors_structure,
                                            avoid_aggregation = avoid_aggregation,
                                            nb_of_neighbors_to_fill = nb_of_neighbors_to_fill,
                                            proceed_even_if_no_probability = proceed_even_if_no_probability)
        
        j_to_exclude += j_allocated
        
        S += s
        
        # if the allocation has been aborted
        if s == 0:
            n_g[v] += 1
        
        N_v[list_v__u.index(v)] -= s
        N_v_allocated[list_v__u.index(v)] += s
    
    
    return(luc_data)


        
        

# def try_land_allocation()
    
    # patch parameters
    # areas = np.zeros(J_pivot.size)
    # eccentricity_mean = {}
    # eccentricity_std = {}
    
    # for state_v in palette_v:
    #     if state_v != state:
    #         j = V_pivot == state_v.value
            
    #         N_j  = j.sum()
            
    #         if N_j > 0:
    #             n_try = 0
    #             area_sum = 0
                
    #             best_id_patches_parameters = None
    #             min_relative_error = np.inf
                
    #             while n_try <= n_patches_tries:
    #                 areas, eccentricities = transition_patches_isl_exp[state_v].draw(N_j)
                    
    #                 area_sum = areas.sum()
    #                 relative_error = np.abs(N_v[id_v] - area_sum) / N_v[id_v]
                    
    #                 if relative_error < min_relative_error:
    #                     min_relative_error = relative_error
    #                     best_id_patches_parameters = id_patches_parameters
                    
    #                 n_try += 1
                
    #             areas[j] = calibrator._patches[u][v]['area'][best_id_patches_parameters]
                
    #             eccentricity_mean[v] = np.mean(calibrator._patches[u][v]['eccentricity'])
    #             eccentricity_std[v] = np.std(calibrator._patches[u][v]['eccentricity'])
                
    
    # # initialize probabilities maps
    # map_P_v__Y = {}
    # for id_v, v in enumerate(list_v__u):
    #     map_P_v__Y[v] = np.zeros(allocated_map.shape)
    #     map_P_v__Y[v].flat[J_Y] = P_v__Y[:, id_v]
        
    #     if 'homogen_proba' in calibrator._patches[u][v].keys():
    #         if calibrator._patches[u][v]['homogen_proba']:
    #             map_P_v__Y[v].fill(1)
            
        
    #     # if it is the first loop, and it is asked to
    #     # save a probability map for each transition
    #     if cnt_loop == 1 and path_prefix_proba_map is not None:
    #         # the probability is re-computed with no patches considerations
    #         # i.e. mono pixel patches (mmp)
    #         P_v__Y_mmp = _compute_P_v__Y(P_v, P_Y, P_Y__v[id_J_to_eval], list_v__u)
    #         map_P_v__Y_mmp = np.zeros(allocated_map.shape)
    #         map_P_v__Y_mmp.flat[J_Y] = P_v__Y_mmp[:, id_v]
            
    #         FeatureLayer(data = map_P_v__Y_mmp,
    #                      copy_geo = calibrator.start_luc_layer,
    #                      path = path_prefix_proba_map+'_v'+str(v)+'.tif')
    
    # # patcher
    # S = 0
    
    # n_g = {v:0 for v in list_v__u}
    
    # j_to_exclude = []
    
    # for id_j in np.random.choice(J_pivot.size, J_pivot.size, replace=False):
        
    #     v = V_pivot[id_j]
    #     s, j_allocated = _weighted_neighbors(map_i_data = region_start_map,
    #                                         map_f_data = allocated_map,
    #                                         map_P_vf__vi_z = map_P_v__Y[v],
    #                                         j_kernel = J_pivot[id_j],
    #                                         vi = u,
    #                                         vf = v,
    #                                         patch_S = areas[id_j],
    #                                         eccentricity_mean = eccentricity_mean[v],
    #                                         eccentricity_std = eccentricity_std[v],
    #                                         neighbors_structure = neighbors_structure,
    #                                         avoid_aggregation = avoid_aggregation,
    #                                         nb_of_neighbors_to_fill = nb_of_neighbors_to_fill,
    #                                         proceed_even_if_no_probability = proceed_even_if_no_probability)
        
    #     j_to_exclude += j_allocated
        
    #     S += s
        
    #     # if the allocation has been aborted
    #     if s == 0:
    #         n_g[v] += 1
        
    #     N_v[list_v__u.index(v)] -= s
    #     N_v_allocated[list_v__u.index(v)] += s
    
    # # patching over
    # # now check if one transition is a total success
    # # if yes, this transition is excluded
    # for v in list_v__u:
    #     if n_g[v] == 0:
    #         N_v[list_v__u.index(v)] = 0
    
    # # check if all transitions are realized
    # if np.all(N_v == 0):
    #     # the process is over
    #     loop = False
    
    # else:
    #     # else, elements set are update
    #     # i.e. allocated elements are removed
    #     id_J_to_eval = ~np.isin(J_Y, j_to_exclude)
    
    # print('target_rest', N_v)
    # print(np.unique(allocated_map.flat[J_Y], return_counts=True))
        
    #     # loop=False
    
    # print('allocated :', N_v_allocated)
    # return(allocated_map)