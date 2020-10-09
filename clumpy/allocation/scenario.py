"""
Scenario
"""

import numpy as np
import pandas as pd
from copy import deepcopy

import os

def adjust_transition_probabilities(tp, P_vf__vi, inplace=False):
    if not inplace:
        tp = deepcopy(tp)
    
    old_P_vf__vi = compute_P_vf__vi_from_transition_probabilities(tp)
    
    for vi in tp.keys():
        tp[vi] = _adjust_transition_probabilities_vi(tp[vi], P_vf__vi[vi], old_P_vf__vi[vi])
            
    if not inplace:
        return(tp)

def _adjust_transition_probabilities_vi(tp_vi, P_vf__vix, old_P_vf__vix, m=1):
    tp_vi = tp_vi * P_vf__vix /  old_P_vf__vix
        
    s = tp_vi.sum(axis=1)
    idx = s > m
    tp_vi[idx, :] = tp_vi[idx, :] / s[idx, None] * m
    
    return(tp_vi)

def force_adjust_transition_probabilities(tp, P_vf__vi, step, epsilon, n_iter=1000, sound=0, inplace=False):
    if not inplace:
        tp = deepcopy(tp)
        
    for vi in tp.keys():
        if sound > 0:
            print('vi='+str(vi))
        
        P_vf__vi_command_vi = P_vf__vi[vi].copy()
        P_vf__vi_returned_vi = _compute_P_vf__vi_from_transition_probabilities_vi(tp[vi])
        
        i = 0
        while((np.abs(P_vf__vi[vi] - P_vf__vi_returned_vi) > epsilon[vi]).sum() > 0):
            i += 1
            if i > n_iter:
                print('maximum number of iterations reached.')
            coef = np.zeros_like(P_vf__vi_command_vi)
            coef[P_vf__vi[vi] - P_vf__vi_returned_vi > epsilon[vi]] = 1
            coef[P_vf__vi[vi] - P_vf__vi_returned_vi < - epsilon[vi]] = -1
            
            P_vf__vi_command_vi = P_vf__vi_command_vi + coef * step[vi]
            
            tp[vi] = _adjust_transition_probabilities_vi(tp[vi], P_vf__vi_command_vi, P_vf__vi_returned_vi)
            
            P_vf__vi_returned_vi = _compute_P_vf__vi_from_transition_probabilities_vi(tp[vi])
        
        if sound>0:
            print('   achieved in '+str(i)+' iterations.')
        
    if not inplace:
        return(tp)

def compute_P_vf__vi_from_transition_probabilities(tp):
    P_vf__vi = {}
    for vi in tp.keys():
        P_vf__vi[vi] = tp[vi].sum(axis=0) / tp[vi].shape[0]
    return(P_vf__vi)

def _compute_P_vf__vi_from_transition_probabilities_vi(tp_vi):
    return(tp_vi.sum(axis=0) / tp_vi.shape[0])

# def compute_multiple_step_transition_matrix(P_vf__vi, nb_steps):
    
#     max_v = int(np.max([P_vf__vi.v.i.max(),np.max(P_vf__vi['P_vf__vi'].columns.to_list())]))
    
#     transition_matrix = np.diag(np.ones(max_v+1))
    
#     for vi in P_vf__vi.v.i.values:
#         for vf in P_vf__vi['P_vf__vi'].columns.to_list():
#             transition_matrix[int(vi),int(vf)] = P_vf__vi.loc[(P_vf__vi.v.i==vi), ('P_vf__vi', vf)].values[0]
    
#     eigen_values, P = np.linalg.eig(transition_matrix)
#     # print(eigen_values)
#     eigen_values = np.power(eigen_values, 1/nb_steps)
    
#     transition_matrix_multiple = np.dot(np.dot(P, np.diag(eigen_values)), np.linalg.inv(P))
#     # return(transition_matrix_multiple)
#     P_vf__vi_multiple = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('v','i')]))
    
#     for vi in P_vf__vi.v.i.values:
#         P_vf__vi_multiple.loc[P_vf__vi_multiple.index.size, ('v','i')] = vi
#         for vf in P_vf__vi['P_vf__vi'].columns.to_list():
#             P_vf__vi_multiple.loc[P_vf__vi_multiple.v.i==vi,('P_vf__vi', vf)] = transition_matrix_multiple[int(vi),int(vf)]
        
#     return(P_vf__vi_multiple)

# def export_transition_matrix_as_dinamica(P_vf__vi, path):
#     df = pd.DataFrame(columns=['From*', 'To*','Rate'])
    
#     for vi in P_vf__vi.v.i.unique():
#         for vf in P_vf__vi.P_vf__vi.columns.to_list():
#             if vf != vi:
#                 if P_vf__vi.loc[P_vf__vi.v.i==vi, ('P_vf__vi', vf)].values[0] > 0:
#                     df.loc[df.index.size] = [int(vi), int(vf), P_vf__vi.loc[P_vf__vi.v.i==vi, ('P_vf__vi', vf)].values[0]]
#                 else:
#                     df.loc[df.index.size] = [int(vi), int(vf), 0]
    
#     # create folder if not exists
#     folder_name = os.path.dirname(path)
#     if not os.path.exists(folder_name) and folder_name!= '':
#         os.makedirs(folder_name)
    
#     df.to_csv(path, index=False)