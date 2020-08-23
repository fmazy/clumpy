import pandas as pd

from .. import definition

import numpy as np

class _Calibration():
    # def __init__(self):
        # self.a = None
    
    def _compute_P_vf__vi_from_P_vf__vi_z(self, J):
        P_vf__vi = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('v','i')]))
    
        P_vf__vi[('v','i')] = np.sort(J.v.i.unique()) 
        
        for vi in np.sort(J.v.i.unique()):
            for vf in J.P_vf__vi_z.columns.to_list():
                P_vf__vi.loc[P_vf__vi.v.i==vi, ('P_vf__vi', vf)] = J.loc[J.v.i==vi, ('P_vf__vi_z', vf)].sum() / J.loc[J.v.i==vi].index.size
                
        return(P_vf__vi)

    def _compute_P_vf__vi(self, case:definition.Case, name='P_vf__vi'):       
        P_vf__vi = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('v','i')]))
        
        df = case.J.groupby([('v','i'), ('v','f')]).size().reset_index(name=('N_vi_vf',''))
        
        for vi in df.v.i.unique():
            df.loc[df.v.i==vi, ('P_vf__vi','')] = df.loc[df.v.i==vi, 'N_vi_vf']/ df.loc[df.v.i==vi, 'N_vi_vf'].sum()
            
            P_vf__vi.loc[P_vf__vi.index.size, ('v','i')] = vi
            for vf in df.loc[df.v.i==vi].v.f.unique():
                P_vf__vi.loc[P_vf__vi.v.i==vi,('P_vf__vi', vf)] = df.loc[(df.v.i==vi) & (df.v.f==vf)].P_vf__vi.values[0]
                
        self.P_vf__vi = P_vf__vi
        self.P_vf__vi.fillna(0, inplace=True)
        
    def compute_P_z__vi(self, case:definition.Case=None, name='P_z__vi', J=None, keep_N=False, output='self'):
        
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
    
    def compute_P_vf__vi_z(self, case:definition.Case=None, J=None, name='P_vf__vi_z', keep_N=False, output='self'):
        """
        Computes transition probabilities directly from entries as simple statistic probabilities.

        Parameters
        ----------
        case : definition.Case (default=``None``)
            The case from witch pixels states and corresponding features are used (``case.discrete_J``). If ``None``, ``J`` is expected.
        J : pandas dataframe (default=``None``)
            The pixels states and corresponding features to use. Expected if ``case=None``.
        name : string (default=``P_vf__vi_z``)
            The output transition probability column name.
        keep_N : boolean (default=``False``)
            If ``True``, the number of concerned pixels is kept for each combination.
        output : {'self', 'return'}, (default=``self``)
            The way to return the result.
            
                self
                    The result is saved as an attribute in ``case.P_vf__vi_z``.
                
                return
                    The result is returned.

        Returns
        -------
        A pandas dataframe which can be returned or saved as an attribute according to ``output``.

        """
        if type(J)==type(None):
            J = case.discrete_J.copy()
        
        col_vi = [('v', 'i')]
        col_vf = [('v','f')]
        cols_z = J[['z']].columns.to_list()
        
        J = J.fillna(-1)
                
        N_z_vi_vf = J.groupby(col_vi+col_vf+cols_z).size().reset_index(name=('N_z_vi_vf','all_vf'))
        N_z_vi = J.groupby(col_vi+cols_z).size().reset_index(name=('N_z_vi',''))
        
        N_z_vi_vf = N_z_vi_vf.merge(N_z_vi, how='left')
        
        list_vf = np.sort(N_z_vi_vf.v.f.unique())
        for vf in list_vf:
            N_z_vi_vfx = N_z_vi_vf.loc[N_z_vi_vf.v.f == vf].copy()
            N_z_vi_vfx[(name,vf)] = N_z_vi_vfx.loc[N_z_vi_vfx.v.f==vf].N_z_vi_vf.all_vf / N_z_vi_vfx.loc[N_z_vi_vfx.v.f==vf].N_z_vi
            
            N_z_vi = N_z_vi.merge(N_z_vi_vfx[col_vi+cols_z+[(name,vf)]], how='left')
        
        if not keep_N:
            N_z_vi.drop(['N_z_vi'], axis=1, level=0, inplace=True)
        
        N_z_vi.fillna(0, inplace=True)
        
        for c in N_z_vi[['z']].columns.to_list():
            N_z_vi.loc[N_z_vi[c] == -1, c] = np.nan
        
        if output=='self':
            self.P_vf__vi_z = N_z_vi
        elif output=='return':
            return(N_z_vi)

    def get_features_and_targets(self, P_vf__vi_z=None):
        """
        Get features and targets of a transition probabilities dataframe.

        Parameters
        ----------
        P_vf__vi_z : pandas dataframe (default=``None``)
            Transition probabilities. If ``None``, ``self.P_vf__vi_z`` is used.
        
        Returns
        -------
        ``X`` the initial state and the features and ``Y`` the targets, i.e. the transitions probabilities
        """
        
        if type(P_vf__vi_z)==type(None):
            P_vf__vi_z = self.P_vf__vi_z.copy()
        
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
        