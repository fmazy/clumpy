"""
Scenario
"""

import numpy as np
import pandas as pd
from copy import deepcopy
import os

from .. import calibration

def compute_transition_probabilities(case,
                                     unique_Z,
                                     P_z__vi_vf,
                                     P_vf__vi,
                                     epsilon=0.1,
                                     n_iter_max=100,
                                     sound=0):
    print('Compute transition probabilities')
    print('\t epsilon='+str(epsilon))
    print('\t n_iter_max='+str(n_iter_max))
    
    unique_Z = deepcopy(unique_Z)
    
    all_Z = case.get_z_as_dataframe()
    
    P_z__vi = calibration.compute_P_z__vi(case)
    
    for vi in P_z__vi_vf.keys():
        if sound > 0:
            print('vi='+str(vi))
        unique_Z[vi] = unique_Z[vi].merge(P_z__vi[vi], how='left')
        
        for id_vf, vf in enumerate(case.dict_vi_vf[vi]):
            unique_Z[vi][('P_z__vi_vf', vf)] = P_z__vi_vf[vi][:, id_vf]
            unique_Z[vi][('P_vf__vi_z', vf)] = 0
        
        n = 0
        achieved = False
        while not achieved and n < n_iter_max:
            n += 1
            if sound > 0:
                print('iteration #'+str(n))
            s = (unique_Z[vi].P_z__vi_vf.values * P_vf__vi[vi]).sum(axis=1) / unique_Z[vi].P_z__vi.values
            
            idx_s = s > 1
            
            # edit P_z__vi_vf
            unique_Z[vi].loc[idx_s, [('P_z__vi_vf', vf) for vf in case.dict_vi_vf[vi]]] = unique_Z[vi].loc[idx_s].P_z__vi_vf.values / s[idx_s, None]
            
            # compute P_vf__vi_z
            unique_Z[vi].P_vf__vi_z = unique_Z[vi].P_z__vi_vf.values * P_vf__vi[vi] / unique_Z[vi].P_z__vi.values[:,None]
                       
            # merge on all pixels
            all_Z[vi] = all_Z[vi][['z']].merge(unique_Z[vi][['z', 'P_vf__vi_z']], how='left')
            
            # check if the scenario is reached
            p = all_Z[vi].P_vf__vi_z.values.mean(axis=0)
            
            if sound > 0:
                print(p / P_vf__vi[vi])
            if (p / P_vf__vi[vi]).min() >= 1 - epsilon:
                achieved = True
            else:
                unique_Z[vi].P_z__vi_vf = unique_Z[vi].P_z__vi_vf / unique_Z[vi].P_z__vi_vf.sum()
    
        all_Z[vi] = all_Z[vi].P_vf__vi_z.values
        
        if sound>0:
            print('achieved for vi='+str(vi))
            print('final P_vf__vi :', p)
            print('=====\n')
        
    return(all_Z)
    
# def adjust_transition_probabilities(tp, P_vf__vi, inplace=False):
#     if not inplace:
#         tp = deepcopy(tp)
    
#     old_P_vf__vi = compute_P_vf__vi_from_transition_probabilities(tp)
    
#     for vi in tp.keys():
#         tp[vi] = _adjust_transition_probabilities_vi(tp[vi], P_vf__vi[vi], old_P_vf__vi[vi])
            
#     if not inplace:
#         return(tp)

# def _adjust_transition_probabilities_vi(tp_vi, P_vf__vix, old_P_vf__vix, m=1):
#     tp_vi = tp_vi * P_vf__vix /  old_P_vf__vix
        
#     s = tp_vi.sum(axis=1)
#     idx = s > m
#     tp_vi[idx, :] = tp_vi[idx, :] / s[idx, None] * m
    
#     return(tp_vi)

# def force_adjust_transition_probabilities(tp, P_vf__vi, step, epsilon, n_iter=1000, sound=0, inplace=False):
#     if not inplace:
#         tp = deepcopy(tp)
        
#     for vi in tp.keys():
#         if sound > 0:
#             print('vi='+str(vi))
        
#         P_vf__vi_command_vi = P_vf__vi[vi].copy()
#         P_vf__vi_returned_vi = _compute_P_vf__vi_from_transition_probabilities_vi(tp[vi])
        
#         i = 0
#         while((np.abs(P_vf__vi[vi] - P_vf__vi_returned_vi) > epsilon[vi]).sum() > 0):
#             i += 1
#             if i > n_iter:
#                 print('maximum number of iterations reached.')
#             coef = np.zeros_like(P_vf__vi_command_vi)
#             coef[P_vf__vi[vi] - P_vf__vi_returned_vi > epsilon[vi]] = 1
#             coef[P_vf__vi[vi] - P_vf__vi_returned_vi < - epsilon[vi]] = -1
            
#             P_vf__vi_command_vi = P_vf__vi_command_vi + coef * step[vi]
            
#             tp[vi] = _adjust_transition_probabilities_vi(tp[vi], P_vf__vi_command_vi, P_vf__vi_returned_vi)
            
#             P_vf__vi_returned_vi = _compute_P_vf__vi_from_transition_probabilities_vi(tp[vi])
        
#         if sound>0:
#             print('   achieved in '+str(i)+' iterations.')
        
#     if not inplace:
#         return(tp)

def compute_P_vf__vi_from_transition_probabilities(tp):
    P_vf__vi = {}
    for vi in tp.keys():
        P_vf__vi[vi] = tp[vi].sum(axis=0) / tp[vi].shape[0]
    return(P_vf__vi)
    
def _compute_P_vf__vi_from_transition_probabilities_vi(tp_vi):
    return(tp_vi.sum(axis=0) / tp_vi.shape[0])

def compute_multi_step_P_vf__vi(P_vf__vi, dict_vi_vf, n):
    
    max_v = int(np.max([np.max(d) for d in dict_vi_vf.values()]))
        
    transition_matrix = np.diag(np.ones(max_v+1))
    
    for vi in dict_vi_vf.keys():
        for id_vf, vf in enumerate(dict_vi_vf[vi]):
            transition_matrix[int(vi),int(vf)] = P_vf__vi[vi][id_vf]
            transition_matrix[int(vf),int(vf)] -= P_vf__vi[vi][id_vf]
    
    transition_matrix = transition_matrix.astype(float)
        
    eigen_values, P = np.linalg.eig(transition_matrix)
    
    
    # print(eigen_values)
    eigen_values = np.power(eigen_values, 1/n)
    
    transition_matrix_multiple = np.dot(np.dot(P, np.diag(eigen_values)), np.linalg.inv(P))
    
    P_vf__vi_multiple = {}
    for vi in dict_vi_vf.keys():
        P_vf__vi_multiple[vi] = []
        for id_vf, vf in enumerate(dict_vi_vf[vi]):
            P_vf__vi_multiple[vi].append(transition_matrix_multiple[vi][vf])
        P_vf__vi_multiple[vi] = np.array(P_vf__vi_multiple[vi])
            
    return(P_vf__vi_multiple)

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