"""
intro
"""

from .. import definition

import pandas as pd
import numpy as np
from .. import tools

def l2_distance(J1,J2):
    
    J1=J1.fillna(-1)
    J2=J2.fillna(-1)
    
    # J = J1.join(other=J2.vf,
    #              how='left',
    #              lsuffix='_1',
    #              rsuffix='_2')
    
    N_vi_vf_1 = J1.groupby(['vi','vf']).size().reset_index(name='N_vi_vf_1')
    N_vi_vf_2 = J2.groupby(['vi','vf']).size().reset_index(name='N_vi_vf_2')
    
    N_vi_vf_z_1 = J1.groupby(list(J1.columns)).size().reset_index(name='N_vi_vf_z_1')
    N_vi_vf_z_2 = J2.groupby(list(J2.columns)).size().reset_index(name='N_vi_vf_z_2')
    
    df = N_vi_vf_z_1.merge(right=N_vi_vf_z_2,
                           how='outer')
    
    df = df.merge(right=N_vi_vf_1,
                  how='left',
                  on=['vi','vf'])
    df = df.merge(right=N_vi_vf_2,
                  how='left',
                  on=['vi','vf'])
    
    df.fillna(0, inplace=True)
    
    df['P_vf__vi_z_1'] = df.N_vi_vf_z_1 / df.N_vi_vf_1
    df['P_vf__vi_z_2'] = df.N_vi_vf_z_2 / df.N_vi_vf_2
    
    d = np.sqrt(np.sum(np.power(df.P_vf__vi_z_1-df.P_vf__vi_z_2,2)))
       
    return(d)

def cramers_V(J,J_ref):
    J=J.fillna(-1)
    J_ref=J_ref.fillna(-1)
        
    N_vi_vf = J.groupby(['vi','vf']).size().reset_index(name='N_vi_vf')
    N_vi_vf_ref = J_ref.groupby(['vi','vf']).size().reset_index(name='N_vi_vf_ref')
    
    N_vi_vf_z = J.groupby(list(J.columns)).size().reset_index(name='N_vi_vf_z')
    N_vi_vf_z_ref = J_ref.groupby(list(J_ref.columns)).size().reset_index(name='N_vi_vf_z_ref')
    
    df = N_vi_vf_z.merge(right=N_vi_vf_z_ref,
                           how='outer')
    
    df = df.merge(right=N_vi_vf,
                  how='left',
                  on=['vi','vf'])
    df = df.merge(right=N_vi_vf_ref,
                  how='left',
                  on=['vi','vf'])
    
    df.fillna(0, inplace=True)
    
    df['P_vf__vi_z'] = df.N_vi_vf_z / df.N_vi_vf
    df['P_vf__vi_z_ref'] = df.N_vi_vf_z_ref / df.N_vi_vf_ref
    
    # restrict to not nul references
    df = df.loc[df.N_vi_vf_z_ref>0]
    
    V = tools.cramers_V(df['P_vf__vi_z'], df['P_vf__vi_z_ref'])
    
    return(V)
    
    