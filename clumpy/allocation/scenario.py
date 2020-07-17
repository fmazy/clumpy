"""
Scenario
"""

import numpy as np
import pandas as pd

import os

def compute_multiple_step_transition_matrix(P_vf__vi, nb_steps):
    
    max_v = int(np.max([P_vf__vi.v.i.max(),np.max(P_vf__vi['P_vf__vi'].columns.to_list())]))
    
    transition_matrix = np.diag(np.ones(max_v+1))
    
    for vi in P_vf__vi.v.i.values:
        for vf in P_vf__vi['P_vf__vi'].columns.to_list():
            transition_matrix[int(vi),int(vf)] = P_vf__vi.loc[(P_vf__vi.v.i==vi), ('P_vf__vi', vf)].values[0]
    
    eigen_values, P = np.linalg.eig(transition_matrix)
    # print(eigen_values)
    eigen_values = np.power(eigen_values, 1/nb_steps)
    
    transition_matrix_multiple = np.dot(np.dot(P, np.diag(eigen_values)), np.linalg.inv(P))
    # return(transition_matrix_multiple)
    P_vf__vi_multiple = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('v','i')]))
    
    for vi in P_vf__vi.v.i.values:
        P_vf__vi_multiple.loc[P_vf__vi_multiple.index.size, ('v','i')] = vi
        for vf in P_vf__vi['P_vf__vi'].columns.to_list():
            P_vf__vi_multiple.loc[P_vf__vi_multiple.v.i==vi,('P_vf__vi', vf)] = transition_matrix_multiple[int(vi),int(vf)]
        
    return(P_vf__vi_multiple)

def export_transition_matrix_as_dinamica(P_vf__vi, path):
    df = pd.DataFrame(columns=['From*', 'To*','Rate'])
    
    for vi in P_vf__vi.v.i.unique():
        for vf in P_vf__vi.P_vf__vi.columns.to_list():
            if vf != vi:
                if P_vf__vi.loc[P_vf__vi.v.i==vi, ('P_vf__vi', vf)].values[0] > 0:
                    df.loc[df.index.size] = [int(vi), int(vf), P_vf__vi.loc[P_vf__vi.v.i==vi, ('P_vf__vi', vf)].values[0]]
                else:
                    df.loc[df.index.size] = [int(vi), int(vf), 0]
    
    # create folder if not exists
    folder_name = os.path.dirname(path)
    if not os.path.exists(folder_name) and folder_name!= '':
        os.makedirs(folder_name)
    
    df.to_csv(path, index=False)