"""
intro
"""

import pandas as pd
import numpy as np

from ..definition import layer
# from .. import tools

def build_P_z__vi_map(J, P_z__vi, map_shape, P_name='P_z__vi'):
    J_with_P = J.reset_index().merge(right=P_z__vi,
                                     how='left').set_index('index')
    
    M = np.zeros(map_shape)
    M.flat[J_with_P.index.values] = J_with_P[P_name].values
    
    return(M)

def build_probability_maps(J, T, P, map_shape, P_name = 'P_vf__vi_z'):
    # check P_vf__vi_z integrity :
    if P_name == 'P_vf__vi_z':
        if P.P_vf__vi_z.sum(axis=1).max() > 1:
            print('warning, max(sum_z(P_vf__vi_z))=',P.P_vf__vi_z.sum(axis=1).max())
        else:
            print('check P_vf__vi_z ok')
    
    probability_maps = layer.LayersP_vf__vi_z()
    
    J_with_P = J.reset_index().merge(right=P,
                                     how='left').set_index('index')
    
    for Ti in T.Ti.values():
        for Tif in Ti.Tif.values():
            probability_map_data = np.zeros(map_shape)
            probability_map_data.flat[J_with_P.index.values] = J_with_P[(P_name, Tif.vf)].values
            
            probability_maps.add_layer(Ti.vi, Tif.vf, probability_map_data)
        
    return(probability_maps)

def computes_P_z__vi(J, name='P_z__vi', keep_N=False):
      
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
    
    return(P_z__vi)
    
def computes_P_z__vi_vf(J, name='P_z__vi_vf', keep_N=False):
    J = J.loc[J.v.i != J.v.f]

    J = J.fillna(-1)

    col_vi = [('v', 'i')]
    col_vf = [('v', 'f')]
    
    cols_z = J[['z']].columns.to_list()
    
    P_z__vi_vf = J[col_vi+cols_z].drop_duplicates()
    
    N_z_vi_vf = J.groupby(col_vi+col_vf+cols_z).size().reset_index(name=('N_z_vi_vf', ''))
    
    N_vi_vf = J.groupby(col_vi+col_vf).size().reset_index(name=('N_vi_vf', ''))

    for vf in J.v.f.unique():        
        P_z__vi_vf = P_z__vi_vf.merge(N_z_vi_vf.rename(columns={'':vf}, level=1).loc[N_z_vi_vf.v.f == vf, col_vi+cols_z+[('N_z_vi_vf',vf)]],
                                      how='left',
                                      on=col_vi+cols_z)
        
        P_z__vi_vf = P_z__vi_vf.merge(N_vi_vf.rename(columns={'':vf}, level=1).loc[N_vi_vf.v.f == vf, col_vi+[('N_vi_vf',vf)]],
                                      how='left',
                                      on=col_vi)
        
    
        P_z__vi_vf[(name, vf)] = P_z__vi_vf[('N_z_vi_vf', vf)] / P_z__vi_vf[('N_vi_vf', vf)]
    
    if not keep_N:
        P_z__vi_vf.drop(['N_z_vi_vf', 'N_vi_vf'], axis=1, level=0, inplace=True)
    
    P_z__vi_vf = P_z__vi_vf.fillna(0)
    
    P_z__vi_vf = P_z__vi_vf.replace(-1, np.nan)
    
    return(P_z__vi_vf)

def computes_P_vf__vi_z(J, name='P_vf__vi_z', keep_N=False):
    J = J.fillna(-1)

    col_vi = [('v', 'i')]
    col_vf = [('v', 'f')]
    
    cols_z = J[['z']].columns.to_list()
        
    N_z_vi_vf = J.loc[J.v.i != J.v.f].groupby(col_vi+col_vf+cols_z).size().reset_index(name=('N_z_vi_vf', ''))
    
    P_vf__vi_z = J.groupby(col_vi+cols_z).size().reset_index(name=('N_z_vi', ''))

    for vf in J.v.f.unique():        
        P_vf__vi_z = P_vf__vi_z.merge(N_z_vi_vf.rename(columns={'':vf}, level=1).loc[N_z_vi_vf.v.f == vf, col_vi+cols_z+[('N_z_vi_vf',vf)]],
                                      how='left',
                                      on=col_vi+cols_z)
        
    
        P_vf__vi_z[(name, vf)] = P_vf__vi_z[('N_z_vi_vf', vf)] / P_vf__vi_z[('N_z_vi', '')]
        
    P_vf__vi_z = P_vf__vi_z.loc[P_vf__vi_z.P_vf__vi_z.sum(axis=1) > 0]
    
    if not keep_N:
        P_vf__vi_z.drop(['N_z_vi_vf', 'N_z_vi'], axis=1, level=0, inplace=True)
    
    P_vf__vi_z = P_vf__vi_z.fillna(0)
    
    P_vf__vi_z = P_vf__vi_z.replace(-1, np.nan)
    
    return(P_vf__vi_z)

def computes_P_vf__vi_z_from_bayes(P_vf__vi, P_z__vi, P_z__vi_vf, keep_P = False):    
    
    col_vi = [('v','i')]
    
    cols_z = P_z__vi_vf[['v','z']].columns.to_list()
    
    P_vf__vi_z = P_z__vi_vf.merge(P_z__vi, how='left', on=col_vi+cols_z)
    
    P_vf__vi_z = P_vf__vi_z.merge(P_vf__vi.astype(float), how='left', on=col_vi)
    
    for vf in P_z__vi_vf.P_z__vi_vf.columns.to_list():
        P_vf__vi_z[('P_vf__vi_z', vf)] = P_vf__vi_z[('P_vf__vi', vf)] * P_vf__vi_z[('P_z__vi_vf', vf)] / P_vf__vi_z[('P_z__vi', '')]
    
    if not keep_P:
        P_vf__vi_z.drop(['P_vf__vi','P_z__vi_vf', 'P_z__vi'], axis=1, level=0, inplace=True)
    
    return(P_vf__vi_z)

def computes_P_z__vi_under_IEVH(J,P_zk__vi):
    if ('v','f') in J.columns:
        J = J.drop(('v', 'f'), axis=1)
    
    P_z__vi = J.drop_duplicates()
    
    Ti = list(P_zk__vi.groupby(['vi']).size().index)
    Z = list(P_zk__vi.groupby(['vi','Zk_name']).size().index)
    
    # print(Ti)
    # print(Z)
    
    for vi in Ti:
        for Zk in Z:
            if Zk[0] == vi:
                Zk_name = Zk[1]
                cols = [('v', 'i'), ('z', Zk_name), ('P_zk__vi', Zk_name)]
                cols = pd.MultiIndex.from_tuples(cols)
                df = pd.DataFrame(columns=cols)
                df[cols] = P_zk__vi.loc[(P_zk__vi.vi == vi) &
                                        (P_zk__vi.Zk_name == Zk_name), ['vi', 'q', 'P_zk__vi']]
                
                P_z__vi = P_z__vi.merge(df.astype(float), how='left')
                
        
        
        P_z__vi.loc[P_z__vi.v.i == vi, ('P_z__vi', '')] = P_z__vi.loc[P_z__vi.v.i == vi].P_zk__vi.product(axis=1, min_count=1)
                
        P_z__vi.drop('P_zk__vi', axis=1, level=0, inplace=True)
    
    # P_z__vi.fillna(0, inplace=True)
    
    return(P_z__vi)

def computes_P_z__vi_vf_under_IEVH(J,P_zk__vi_vf):
    if ('v','f') in J.columns:
        J = J.drop(('v', 'f'), axis=1)
    
    P_z__vi_vf = J.drop_duplicates()
    
    T = list(P_zk__vi_vf.groupby(['vi','vf']).size().index)
    Z = list(P_zk__vi_vf.groupby(['vi','Zk_name']).size().index)
    
    # print(T)
    # print(Z)
    
    for Tk in T:
        vi = Tk[0]
        vf = Tk[1]
        for Zk in Z:
            if Zk[0] == vi:
                Zk_name = Zk[1]
                cols = [('v', 'i'), ('z', Zk_name), ('P_zk__vi_vf', Zk_name)]
                cols = pd.MultiIndex.from_tuples(cols)
                df = pd.DataFrame(columns=cols)
                df[cols] = P_zk__vi_vf.loc[(P_zk__vi_vf.vi == vi) &
                                                 (P_zk__vi_vf.vf == vf) &
                                                 (P_zk__vi_vf.Zk_name == Zk_name), ['vi', 'q', 'P_zk__vi_vf']]
                
                P_z__vi_vf = P_z__vi_vf.merge(df.astype(float), how='left')
        
        P_z__vi_vf.loc[P_z__vi_vf.v.i==vi, ('P_z__vi_vf', vf)] = P_z__vi_vf.loc[P_z__vi_vf.v.i==vi].P_zk__vi_vf.product(axis=1, min_count=1)
        
        P_z__vi_vf.drop('P_zk__vi_vf', axis=1, level=0, inplace=True)
    
    # P_z__vi_vf.fillna(0, inplace=True)
    
    return(P_z__vi_vf)

