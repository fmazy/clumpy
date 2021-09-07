#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 13:51:30 2021

@author: frem
"""

import numpy as np
from ._generalized_allocation import generalized_allocation_rejection_test
from ._patcher import _weighted_neighbors
from ..definition import LandUseCoverLayer, FeatureLayer
from ..calibration._calibrator import _compute_P_v__Y

def generic_allocator(calibrators,
                      tms,
                      path,
                      n_patches_tries = 10**3,
                      update_P_Y = True,
                      path_prefix_proba_map = None,
                      patcher_params = None):
    
    start_luc_layer = calibrators[0].start_luc_layer
    allocated_map = start_luc_layer.get_data().copy()
    
    for id_calibrator, calibrator in enumerate(calibrators):
        print('=============================')
        print('Region #'+str(id_calibrator))
        print('=============================')
        
        tm = tms[id_calibrator]
        
        allocated_map = _generic_allocator_region_process(calibrator,
                                                        allocated_map,
                                                        tm,
                                                        n_patches_tries = n_patches_tries,
                                                        update_P_Y = update_P_Y,
                                                        path_prefix_proba_map = path_prefix_proba_map+'_A'+str(id_calibrator),
                                                        patcher_params = patcher_params)
            
    map_f = LandUseCoverLayer(name="allocated map",
                              path = path, 
                              copy_geo=start_luc_layer,
                              data = allocated_map)
    
    return(map_f)

def _generic_allocator_region_process(calibrator,
                                      allocated_map,
                                      tm,
                                      n_patches_tries = 10**3,
                                      update_P_Y=True,
                                      path_prefix_proba_map=None,
                                      patcher_params = None):
    
    for u in calibrator._calibrated_transitions_u.keys():
        print('----------------')
        print('u='+str(u))
        print('----------------')
        allocated_map = _generic_allocator_region_process_u_fixed(calibrator = calibrator,
                                              allocated_map = allocated_map,
                                              tm = tm,
                                              u = u,
                                              n_patches_tries = n_patches_tries,
                                              update_P_Y = update_P_Y,
                                              path_prefix_proba_map = path_prefix_proba_map+'_u'+str(u),
                                              patcher_params = patcher_params)

    return(allocated_map)

def _generic_allocator_region_process_u_fixed(calibrator,
                                              allocated_map,
                                              tm,
                                              u,
                                              n_patches_tries = 10**3,
                                              update_P_Y=True,
                                              cnt_loop_max=500,
                                              path_prefix_proba_map=None,
                                              patcher_params = None):
    
    if patcher_params is not None:
        neighbors_structure = patcher_params['neighbors_structure']
        avoid_aggregation = patcher_params['avoid_aggregation']
        nb_of_neighbors_to_fill = patcher_params['nb_of_neighbors_to_fill']
        proceed_even_if_no_probability = patcher_params['proceed_even_if_no_probability']
    
    else:
        neighbors_structure = 'rook'
        avoid_aggregation = True
        nb_of_neighbors_to_fill = 3
        proceed_even_if_no_probability = True
    
    # set the sub starting map
    # it will be used to check allocated patches
    region_start_map = allocated_map.copy() * calibrator.region_eval.get_data()
        
    # get global transition probabilities
    P_v, list_v__P_v = tm.P_v(u=u)
    
    # size reduction to calibrated transition probabilities
    list_v__u = calibrator._calibrated_transitions_u[u]
    P_v = np.array([P_v[list_v__P_v.index(v)] for v in list_v__u])
    
    # get elements of the initial state
    J_Y = calibrator._J_Y_u[u].copy()
    Y = calibrator._Y_u[u].copy()
    
    # set the target volume
    N = J_Y.size
    N_v = P_v * N
    
    N_v_allocated = np.zeros(N_v.size)
    
    print('target for u='+str(u), N_v)
    
    # if no allocations are expected
    if np.isclose(N_v.sum(), 0):
        print('no allocations expected.')
        return(allocated_map)
    
    # compute P(Y|v, u) (u is omitted)
    # it is computed only one time
    P_Y__v, list_v__P_Y__v = calibrator._estimate_P_Y__v(Y, u, log_return=False)
    
    id_J_to_eval = np.ones(P_Y__v.shape[0]).astype(bool)
    
    # start main loop
    cnt_loop = 0
    loop = True
    while loop and cnt_loop < cnt_loop_max:
        cnt_loop += 1
        
        # compute P(Y|u)
        # it will be computed at each loop
        # or at least at the first loop is the parameter
        # update_P_Y is False
        if update_P_Y or cnt_loop == 1:
            P_Y = calibrator._estimate_P_Y(Y[id_J_to_eval], u, log_return=False)
            
            if not update_P_Y:
                P_Y_full = P_Y.copy()
            
        elif cnt_loop > 1:
            # if update_P_Y is False and it is not the first loop
            # one exclude values of P_Y
            P_Y = P_Y_full[id_J_to_eval]
        
        # set particular P_v for this loop
        P_v = N_v / N
        
        # the P_v is deduced by patch areas mean
        P_v_patches = P_v / np.array([calibrator._patches[u][v]['area'].mean() for v in list_v__u])
                
        P_v__Y = _compute_P_v__Y(P_v_patches, P_Y, P_Y__v[id_J_to_eval], list_v__u)
        
        # GART
        # the probabilities of no-transition are stacked to P
        # to allow no-transition through the GART
        J_pivot, V_pivot = _gart(J_Y, id_J_to_eval, P_v__Y, list_v__u, u)
        
        # patch parameters
        
        areas = np.zeros(J_pivot.size)
        eccentricity_mean = {}
        eccentricity_std = {}
        
        for id_v, v in enumerate(list_v__u):
            j = V_pivot == v
            
            N_j  = j.sum()
            
            if N_j > 0:
                n_try = 0
                area_sum = 0
                
                best_id_patches_parameters = None
                min_relative_error = np.inf
                
                while n_try <= n_patches_tries:
                    id_patches_parameters = np.random.choice(calibrator._patches[u][v]['area'].size,
                                                            size = N_j,
                                                            replace=True)
                    
                    area_sum = calibrator._patches[u][v]['area'][id_patches_parameters].sum()
                    relative_error = np.abs(N_v[id_v] - area_sum) / N_v[id_v]
                    
                    if relative_error < min_relative_error:
                        min_relative_error = relative_error
                        best_id_patches_parameters = id_patches_parameters
                    
                    n_try += 1
                
                areas[j] = calibrator._patches[u][v]['area'][best_id_patches_parameters]
                
                eccentricity_mean[v] = np.mean(calibrator._patches[u][v]['eccentricity'])
                eccentricity_std[v] = np.std(calibrator._patches[u][v]['eccentricity'])
                
            print(str(u)+'->'+str(v)+', N_j='+str(N_j)+', target='+str(N_v[id_v])+', expected area='+str(areas[j].sum()))
        
        # initialize probabilities maps
        map_P_v__Y = {}
        for id_v, v in enumerate(list_v__u):
            map_P_v__Y[v] = np.zeros(allocated_map.shape)
            map_P_v__Y[v].flat[J_Y] = P_v__Y[:, id_v]
            
            if 'homogen_proba' in calibrator._patches[u][v].keys():
                if calibrator._patches[u][v]['homogen_proba']:
                    map_P_v__Y[v].fill(1)
                
            
            # if it is the first loop, and it is asked to
            # save a probability map for each transition
            if cnt_loop == 1 and path_prefix_proba_map is not None:
                # the probability is re-computed with no patches considerations
                # i.e. mono pixel patches (mmp)
                P_v__Y_mmp = _compute_P_v__Y(P_v, P_Y, P_Y__v[id_J_to_eval], list_v__u)
                map_P_v__Y_mmp = np.zeros(allocated_map.shape)
                map_P_v__Y_mmp.flat[J_Y] = P_v__Y_mmp[:, id_v]
                
                FeatureLayer(data = map_P_v__Y_mmp,
                             copy_geo = calibrator.start_luc_layer,
                             path = path_prefix_proba_map+'_v'+str(v)+'.tif')
        
        # patcher
        S = 0
        
        n_g = {v:0 for v in list_v__u}
        
        j_to_exclude = []
        
        for id_j in np.random.choice(J_pivot.size, J_pivot.size, replace=False):
            
            v = V_pivot[id_j]
            s, j_allocated = _weighted_neighbors(map_i_data = region_start_map,
                                                map_f_data = allocated_map,
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
        
        # patching over
        # now check if one transition is a total success
        # if yes, this transition is excluded
        for v in list_v__u:
            if n_g[v] == 0:
                N_v[list_v__u.index(v)] = 0
        
        # check if all transitions are realized
        if np.all(N_v == 0):
            # the process is over
            loop = False
        
        else:
            # else, elements set are update
            # i.e. allocated elements are removed
            id_J_to_eval = ~np.isin(J_Y, j_to_exclude)
        
        print('target_rest', N_v)
        print(np.unique(allocated_map.flat[J_Y], return_counts=True))
        
        # loop=False
    
    print('allocated :', N_v_allocated)
    return(allocated_map)

def _gart(J_Y, id_J_to_eval, P_v__Y, list_v__u, u):
    
    # security
    
    V_pivot = generalized_allocation_rejection_test(np.hstack((P_v__Y, 1 - P_v__Y.sum(axis=1)[:, None])),
                                                  list_v__u + [u])
        
    id_pivot = V_pivot != u
    
    J_pivot = J_Y[id_J_to_eval][id_pivot]
    V_pivot = V_pivot[id_pivot]
    
    return(J_pivot, V_pivot)

def _log(x):
    return(np.log(x,
                  out = np.zeros_like(x).fill(-np.inf),
                  where = x > 0))