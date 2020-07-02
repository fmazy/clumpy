import pandas as pd

from .. import definition
from . import discretization

import numpy as np

class _Calibration():
    # def __init__(self):
        # self.a = None

    def _compute_P_vf__vi(self, case:definition.Case, name='P_vf__vi'):       
        P_vf__vi = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('v','i')]))
        
        df = case.discrete_J.groupby([('v','i'), ('v','f')]).size().reset_index(name=('N_vi_vf',''))
        
        for vi in df.v.i.unique():
            df.loc[df.v.i==vi, ('P_vf__vi','')] = df.loc[df.v.i==vi, 'N_vi_vf']/ df.loc[df.v.i==vi, 'N_vi_vf'].sum()
            
            P_vf__vi.loc[P_vf__vi.index.size, ('v','i')] = vi
            for vf in df.loc[df.v.i==vi].v.f.unique():
                P_vf__vi.loc[P_vf__vi.v.i==vi,('P_vf__vi', vf)] = df.loc[(df.v.i==vi) & (df.v.f==vf)].P_vf__vi.values[0]
        
        self.P_vf__vi = P_vf__vi
        
    def compute_P_z__vi(self, case:definition.Case, name='P_z__vi', J=None, keep_N=False, output='self'):
        
        if type(J)==type(None):
            J = case.discrete_J
        
        J = J.fillna(-1)
        
        col_vi = [('v', 'i')]
        cols_z = J[['z']].columns.to_list()
    
        P_z__vi = J.groupby(col_vi+cols_z).size().reset_index(name=('N_z_vi',''))
        
        N_vi = J.groupby(col_vi).size().reset_index(name=('N_vi', ''))
        
        P_z__vi = P_z__vi.merge(N_vi, how='left', on=col_vi)
        
        P_z__vi[name] = P_z__vi['N_z_vi'] / P_z__vi['N_vi']
        
        if not keep_N:
            P_z__vi.drop(['N_vi', 'N_z_vi'], axis=1, level=0, inplace=True)
        
        P_z__vi.z = P_z__vi.z.replace(-1, np.nan)
        
        if output=='self':
            self.P_z__vi = P_z__vi
        elif output=='return':
            return(P_z__vi)
        
    def build_P_z__vi_map(self, case:definition.Case, P_name='P_z__vi', J=None):
        
        if type(J)==type(None):
            J = case.discrete_J
        
        J_with_P = J.reset_index().merge(right=self.P_z__vi,
                                         how='left').set_index('index')
        
        M = np.zeros(case.map_i.data.shape)
        M.flat[J_with_P.index.values] = J_with_P[P_name].values
        
        return(M)