import pandas as pd

from .. import definition
from ..tools import np_suitable_integer_type

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

class _Calibration():
    # def __init__(self):
        # self.a = None
    
 

        
    # def r2_score()
        
    def build_P_z__vi_map(self, case:definition.Case, P_name='P_z__vi', J=None):
        
        if type(J)==type(None):
            J = case.discrete_J
        
        J_with_P = J.reset_index().merge(right=self.P_z__vi,
                                         how='left').set_index('index')
        
        M = np.zeros(case.map_i.data.shape)
        M.flat[J_with_P.index.values] = J_with_P[P_name].values
        
        return(M)
    

def compute_P_vf__vi_from_P_vf__vi_z(J, name='P_vf__vi'):
    P_vf__vi = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('v','i')]))

    P_vf__vi[('v','i')] = np.sort(J.v.i.unique()) 
    
    for vi in np.sort(J.v.i.unique()):
        for vf in J.P_vf__vi_z.columns.to_list():
            P_vf__vi.loc[P_vf__vi.v.i==vi, (name, vf)] = J.loc[J.v.i==vi, ('P_vf__vi_z', vf)].sum() / J.loc[J.v.i==vi].index.size
            
    return(P_vf__vi)     


def get_features_and_targets(P_vf__vi_z):
    """
    Get features and targets of a transition probabilities dataframe.

    Parameters
    ----------
    P_vf__vi_z : pandas dataframe 
        Transition probabilities. 
    
    Returns
    -------
    ``X`` the initial state and the features and ``Y`` the targets, i.e. the transitions probabilities
    """    
    col_vi = [('v', 'i')]
    cols_z = P_vf__vi_z[['z']].columns.to_list()
    cols_P = P_vf__vi_z[['P_vf__vi_z']].columns.to_list()
    
    X = P_vf__vi_z[col_vi+cols_z]
    Y =  P_vf__vi_z[cols_P]
    
    return(X,Y)
    
    # def transition_probability_maps():
        
    
def _clean_X(X):
    # check if a X column is full of nan:
    columns_sumed_na = np.isnan(X).sum(axis=0)
    for idx, c in enumerate(columns_sumed_na):
        if c == X.shape[0]:
            X = np.delete(X, idx, 1) # delete the column
    return(X)

# def compute_P_z__vi_vf(J=None, name='P_z__vi_vf'):
#     """
#     Computes transition probabilities directly from entries as simple statistic probabilities.

#     Parameters
#     ----------
#     case : definition.Case (default=``None``)
#         The case from witch pixels states and corresponding features are used (``case.discrete_J``). If ``None``, ``J`` is expected.
#     J : pandas dataframe (default=``None``)
#         The pixels states and corresponding features to use. Expected if ``case=None``.
#     name : string (default=``P_vf__vi_z``)
#         The output transition probability column name.
#     keep_N : boolean (default=``False``)
#         If ``True``, the number of concerned pixels is kept for each combination.
#     output : {'self', 'return'}, (default=``self``)
#         The way to return the result.
        
#             self
#                 The result is saved as an attribute in ``case.P_vf__vi_z``.
            
#             return
#                 The result is returned.

#     Returns
#     -------
#     A pandas dataframe which can be returned or saved as an attribute according to ``output``.

#     """        
#     col_vi = [('v', 'i')]
#     col_vf = [('v','f')]
#     cols_z = J[['z']].columns.to_list()
    
#     J = J.fillna(-1)
            
#     N_z_vi_vf = J.groupby(col_vi+col_vf+cols_z).size().reset_index(name=('N_z_vi_vf',''))
#     N_vi_vf = J.groupby(col_vi+col_vf).size().reset_index(name=('N_vi_vf',''))
    
#     N_z_vi_vf = N_z_vi_vf.merge(N_vi_vf, how='left')
    
#     N_z_vi_vf[(name, 'all')] = N_z_vi_vf.N_z_vi_vf / N_z_vi_vf.N_vi_vf
    
#     N_z_vi_vf.drop(['N_z_vi_vf', 'N_vi_vf'], axis=1, level=0, inplace=True)
    
#     P_z__vi_vf = J[col_vi + cols_z].drop_duplicates()
    
#     for vf in np.sort(N_z_vi_vf.v.f.unique()):
#         N_z_vi_vf_to_merge = N_z_vi_vf.loc[N_z_vi_vf.v.f==vf].copy()
#         N_z_vi_vf_to_merge[(name, vf)] = N_z_vi_vf_to_merge[name, 'all']
#         N_z_vi_vf_to_merge.drop([(name, 'all'), ('v','f')], axis=1, inplace=True)
#         P_z__vi_vf = P_z__vi_vf.merge(N_z_vi_vf_to_merge, how='left') 
    
#     P_z__vi_vf[cols_z] = P_z__vi_vf[cols_z].replace(to_replace=-1, value=np.nan).values
    
#     return(P_z__vi_vf)

def compute_P_vi(J, name='P_vi'):
    df = J.groupby([('v','i')]).size().reset_index(name=(name,''))
    df[name] = df[name] / df[name].sum()
    return(df)
    # P_vi = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('v','i'), (name, '')]))
    
    # for vi in J.v.i.unique():
    #     P_vi.loc[P_vi.index.size] = [vi, J.loc[J.v.i.unique()]]

def compute_P_vf__vi(J, name='P_vf__vi'):       
    P_vf__vi = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('v','i')]))
    
    df = J.groupby([('v','i'), ('v','f')]).size().reset_index(name=('N_vi_vf',''))
    
    for vi in df.v.i.unique():
        df.loc[df.v.i==vi, ('P_vf__vi','')] = df.loc[df.v.i==vi, 'N_vi_vf']/ df.loc[df.v.i==vi, 'N_vi_vf'].sum()
        
        P_vf__vi.loc[P_vf__vi.index.size, ('v','i')] = vi
        for vf in df.loc[df.v.i==vi].v.f.unique():
            P_vf__vi.loc[P_vf__vi.v.i==vi,('P_vf__vi', vf)] = df.loc[(df.v.i==vi) & (df.v.f==vf)].P_vf__vi.values[0]
    
    P_vf__vi = P_vf__vi.reindex(sorted(P_vf__vi.columns), axis=1)
    
    return(P_vf__vi.fillna(0))

def compute_N_z_vi(case, name='N_z_vi'):
    N_z_vi = {}
    for vi in case.Z.keys():
        col_names = pd.MultiIndex.from_tuples([('z',z_name) for z_name in case.Z_names[vi]])
        N_z_vi[vi] = pd.DataFrame(case.Z[vi], columns=col_names)
        N_z_vi[vi] = N_z_vi[vi].groupby(by=N_z_vi[vi].columns.to_list()).size().reset_index(name=(name, ''))
    return(N_z_vi)

def compute_N_z_vi_vf(case, name='N_z_vi_vf'):
    N_z_vi_vf = {}
    for vi in case.Z.keys():
        N_z_vi_vf[vi] = {}
        for vf in case.dict_vi_vf[vi]:
            col_names = pd.MultiIndex.from_tuples([('z',z_name) for z_name in case.Z_names[vi]])
            N_z_vi_vf[vi][vf] = pd.DataFrame(case.Z[vi][case.vf[vi] == vf],
                                             columns=col_names)
            N_z_vi_vf[vi][vf] = N_z_vi_vf[vi][vf].groupby(by=N_z_vi_vf[vi][vf].columns.to_list()).size().reset_index(name=(name, vf))
    return(N_z_vi_vf)

def compute_P_z__vi(case, name='P_z__vi'):
    P_z__vi = compute_N_z_vi(case, name=name)
    for vi in case.Z.keys():
        P_z__vi[vi][name] /= case.Z[vi].shape[0]
    return(P_z__vi)

def _distance_to_weights(d):
    w = (np.zeros_like(d) + 1).astype(np.float)
    w[d!=0] = 1/d[d!=0]
    w[w>1] = 1
    return(w)

def compute_P_z__vi_vf(case, name='P_z__vi_vf', n_smooth = 5):
    P_z__vi_vf = compute_N_z_vi_vf(case, name='P_z__vi_vf')
    
    N_z_vi = compute_N_z_vi(case)
    
    for vi in P_z__vi_vf.keys():
        N_z_vi[vi].drop(['N_z_vi'], axis=1, level=0, inplace=True)
        
        for vf in P_z__vi_vf[vi].keys():
            P_z__vi_vf[vi][vf][('P_z__vi_vf', vf)] /= P_z__vi_vf[vi][vf][('P_z__vi_vf', vf)].sum()
            
            N_z_vi[vi] = N_z_vi[vi].merge(right=P_z__vi_vf[vi][vf],
                                          how='left')
            
        N_z_vi[vi].fillna(0, inplace=True)
            
        if n_smooth is not None:
            knr = KNeighborsRegressor(n_neighbors=n_smooth, weights=_distance_to_weights)
            knr.fit(N_z_vi[vi].z.values, N_z_vi[vi].P_z__vi_vf.values)
            
            N_z_vi[vi][['P_z__vi_vf']] = knr.predict(N_z_vi[vi].z.values)
    
    return(N_z_vi)
    
    
def compute_P_vf__vi_z(case, name='P_vf__vi_z'):
    N_z_vi_vf = compute_N_z_vi_vf(case, 'P_vf__vi_z')
    P_vf__vi_z = compute_N_z_vi(case)
    
    for vi in N_z_vi_vf.keys():
        dict_columns_fillna = {}
        for vf in N_z_vi_vf[vi].keys():
            P_vf__vi_z[vi] = P_vf__vi_z[vi].merge(right=N_z_vi_vf[vi][vf],
                                                  how='left')
            dict_columns_fillna[('P_vf__vi_z',vf)] = 0
        
        # fillna
        P_vf__vi_z[vi].fillna(dict_columns_fillna, inplace=True)
        
        # divide
        P_vf__vi_z[vi][['P_vf__vi_z']] = P_vf__vi_z[vi][['P_vf__vi_z']].div(P_vf__vi_z[vi]['N_z_vi'],
                                                                            axis=0).values
    
        # remove N_z_vi column
        P_vf__vi_z[vi].drop(['N_z_vi'], axis=1, level=0, inplace=True)
    
    return(P_vf__vi_z)



def compute_P_vf__vi_z_with_bayes(P_vf__vi, P_z__vi, P_z__vi_vf, keep_P=False):
    
    # cols_z = P_z__vi[['z']].columns.to_list()
    
    # P_z__vi[cols_z] = P_z__vi[cols_z].fillna(-1).values
    # P_z__vi_vf[cols_z] = P_z__vi_vf[cols_z].fillna(-1).values
    
    P_vf__vi_z = P_z__vi_vf.merge(P_z__vi, how='left')
        
    P_vf__vi_z = P_vf__vi_z.merge(P_vf__vi, how='left')
    
    for vf in P_z__vi_vf.P_z__vi_vf.columns.to_list():
        idx = P_vf__vi_z.loc[P_vf__vi_z.P_z__vi > 0].index.values
        P_vf__vi_z.loc[idx, ('P_vf__vi_z', vf)] = P_vf__vi_z.loc[idx, ('P_vf__vi', vf)] * P_vf__vi_z.loc[idx, ('P_z__vi_vf', vf)] / P_vf__vi_z.loc[idx, ('P_z__vi', '')]
    
    if not keep_P:
        P_vf__vi_z.drop(['P_vf__vi','P_z__vi_vf', 'P_z__vi'], axis=1, level=0, inplace=True)
    
    # P_vf__vi_z[cols_z] = P_vf__vi_z[cols_z].replace(to_replace=-1, value=np.nan).values
    
    return(P_vf__vi_z)