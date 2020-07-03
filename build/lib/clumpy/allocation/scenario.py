"""
Scenario
"""

import numpy as np
import pandas as pd

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