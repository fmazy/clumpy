from ._calibration import _Calibration
from ..definition._case import Case
from ..definition._layer import TransitionProbabilityLayers

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class NaiveBayes(_Calibration):
    """
    Naive Bayes calibration method.
    """
    # def __init__(self):
        # super().__init__()
        
    def fit(self, J, binarizer):
        """
        fit model with a discretized case

        Parameters
        ----------
        case : Case
            The case which have to be discretized.

        Notes
        -----
        
        New attributes are availables :
            
            ``self.P_zk__vi``
            
            ``self.P_zk__vi_vf``
            
            ``self.P_vf__vi``

        """
        self.binarizer = binarizer
        self._compute_P_zk__vi(J)
        self._compute_P_zk__vi_vf(J)
        self.P_vf__vi = self._compute_P_vf__vi(J)
        
    def predict(self, J):
        P_z__vi = self._compute_naive_P_z__vi(J).fillna(0)
        P_z__vi_vf = self._compute_naive_P_z__vi_vf(J).fillna(0)
        P_vf__vi_z = _compute_P_vf__vi_z_with_bayes(self.P_vf__vi, P_z__vi, P_z__vi_vf).fillna(0)
        
        return(J.merge(P_vf__vi_z, how='left'))
        
    def transition_probability_maps(self, case:Case, P_vf__vi = None):
        """
        Infer transition probability maps 

        Parameters
        ----------
        case : Case
            The case which have to be discretized..
        P_vf__vi : Pandas DataFrame (default=None)
            The transition matrix. If ``None``, the self computed one is used.
        
        Returns
        -------
        maps : TransitionProbabilityLayers
            The transition probability maps
        """
        if type(P_vf__vi) == type(None):
            self._compute_P_vf__vi(case)
            P_vf__vi = self.P_vf__vi
        
        P_z__vi = self._compute_naive_P_z__vi(case).fillna(0)
        P_z__vi_vf = self._compute_naive_P_z__vi_vf(case).fillna(0)
        
        P_vf__vi_z = _compute_P_vf__vi_z_with_bayes(P_vf__vi, P_z__vi, P_z__vi_vf).fillna(0)
        
        # return(P_vf__vi_z)
        
        # print(P_vf__vi_z)
        
        maps = _build_probability_maps(case, P_vf__vi_z)
        
        return(maps)
        
    def _compute_P_zk__vi(self, J):
        self.P_zk__vi = pd.DataFrame(columns=['vi','Zk_name','q','P_zk__vi'])
        
        for vi in J.v.i.unique():
            # restriction to considered pixels
            J_vi = J.loc[(J.v.i==vi)]
            for Zk_name in J.z.columns.to_list():
                
                # restriction to considered alpha
                alpha_Zk = self.binarizer.alpha[(vi, Zk_name)]
                
                # we count every unique combinaisons of vi, Zk
                count = J_vi.z.groupby([Zk_name]).size().reset_index(name='P_zk__vi')
                
                # we fill holes where no occurences have been found
                q = count[Zk_name].values
                n = count['P_zk__vi'].values
                n_full = np.zeros((len(alpha_Zk)+1))
                n_full[q.astype(int)] = n
                
                # sub df creation
                df_sub = pd.DataFrame(columns=['vi','Zk_name','q','P_zk__vi'])
                df_sub.q = np.arange((len(alpha_Zk)+1))
                df_sub.P_zk__vi = n_full/n_full.sum()
                df_sub['vi'] = vi
                df_sub['Zk_name'] = Zk_name
                
                # self.P_zk__vi concatenation
                self.P_zk__vi = pd.concat([self.P_zk__vi, df_sub], ignore_index=True)
                
    def _compute_P_zk__vi_vf(self, J):        
        self.P_zk__vi_vf = pd.DataFrame(columns=['vi','vf','Zk_name','q','P_zk__vi_vf'])
        
        for vi in J.v.i.unique():
            # restriction to considered pixels
            for Zk_name in J.z.columns.to_list():
                for vf in J.v.f.unique():
                    # restriction to considered pixels
                    J_vi_vf = J.loc[(J.v.i==vi) & (J.v.f==vf)]
                    
                        
                    # restriction to considered alpha
                    alpha_Zk = self.binarizer.alpha[(vi, Zk_name)]
                    
                    # we count every unique combinaisons of vi, vf, Zk
                    count = J_vi_vf.z.groupby([Zk_name]).size().reset_index(name='P_zk__vi_vf')
                    
                    # we fill holes where no occurences have been found
                    q = count[Zk_name].values
                    n = count['P_zk__vi_vf'].values
                    n_full = np.zeros((len(alpha_Zk)+1))
                    n_full[q.astype(int)] = n
                    
                    # sub df creation
                    df_sub = pd.DataFrame(columns=['vi','vf','Zk_name','q','P_zk__vi_vf'])
                    df_sub.q = np.arange((len(alpha_Zk)+1))
                    df_sub.P_zk__vi_vf = n_full / n_full.sum()
                    df_sub['vi'] = vi
                    df_sub['vf'] = vi
                    df_sub['Zk_name'] = Zk_name
                    
                    # self.P_zk__vi_vf concatenation
                    self.P_zk__vi_vf = pd.concat([self.P_zk__vi_vf, df_sub], ignore_index=True)
    
    def _compute_naive_P_z__vi(self, J):
        J = J.copy()
        
        if ('v','f') in J.columns:
            J = J.drop(('v', 'f'), axis=1)
        
        P_z__vi = J.drop_duplicates()
        
        Ti = list(self.P_zk__vi.groupby(['vi']).size().index)
        Z = list(self.P_zk__vi.groupby(['vi','Zk_name']).size().index)
        
        # print(Ti)
        # print(Z)
        
        for vi in Ti:
            for Zk in Z:
                if Zk[0] == vi:
                    Zk_name = Zk[1]
                    cols = [('v', 'i'), ('z', Zk_name), ('P_zk__vi', Zk_name)]
                    cols = pd.MultiIndex.from_tuples(cols)
                    df = pd.DataFrame(columns=cols)
                    df[cols] = self.P_zk__vi.loc[(self.P_zk__vi.vi == vi) &
                                            (self.P_zk__vi.Zk_name == Zk_name), ['vi', 'q', 'P_zk__vi']]
                    
                    P_z__vi = P_z__vi.merge(df.astype(float), how='left')
                    
            
            
            P_z__vi.loc[P_z__vi.v.i == vi, ('P_z__vi', '')] = P_z__vi.loc[P_z__vi.v.i == vi].P_zk__vi.product(axis=1, min_count=1)
                    
            P_z__vi.drop('P_zk__vi', axis=1, level=0, inplace=True)
        
        return(P_z__vi)
    
    def _compute_naive_P_z__vi_vf(self, J):
        J = J.copy()
        
        if ('v','f') in J.columns:
            J = J.drop(('v', 'f'), axis=1)
        
        P_z__vi_vf = J.drop_duplicates()
        
        T = list(self.P_zk__vi_vf.groupby(['vi','vf']).size().index)
        Z = list(self.P_zk__vi_vf.groupby(['vi','Zk_name']).size().index)
        
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
                    df[cols] = self.P_zk__vi_vf.loc[(self.P_zk__vi_vf.vi == vi) &
                                                     (self.P_zk__vi_vf.vf == vf) &
                                                     (self.P_zk__vi_vf.Zk_name == Zk_name), ['vi', 'q', 'P_zk__vi_vf']]
                    
                    P_z__vi_vf = P_z__vi_vf.merge(df.astype(float), how='left')
            
            P_z__vi_vf.loc[P_z__vi_vf.v.i==vi, ('P_z__vi_vf', vf)] = P_z__vi_vf.loc[P_z__vi_vf.v.i==vi].P_zk__vi_vf.product(axis=1, min_count=1)
            
            P_z__vi_vf.drop('P_zk__vi_vf', axis=1, level=0, inplace=True)
                
        return(P_z__vi_vf)
    
    def plot_P_zk__vi(self, vi, Zk_name, max_one=False, sum_one=False, color=None, linestyle='-', linewidth=1.5, label=None, alpha=None, step=True):
        """
        plots a given :math:`P_k(\hat{z}|v_i,v_f)`.
        
        """
        # for colors, see https://matplotlib.org/3.1.0/gallery/color/named_colors.html
        
        P_zk__vi = self.P_zk__vi.loc[(self.P_zk__vi.vi == vi) &
                              (self.P_zk__vi.Zk_name == Zk_name)]
        
        y = P_zk__vi.P_zk__vi.values
        y = y[1:-1]
        
        if type(alpha) == pd.DataFrame:
            x = alpha.loc[(alpha.vi == vi) &
                          (alpha.Zk_name == Zk_name)].alpha.values
            x = x[0:-1]
        else:
            x = P_zk__vi.q.values
            x = x[1:-1]
        
        if sum_one:
            y = y/np.sum(y)
            
        if max_one:
            y = y/np.max(y)
        
        if step:
            plt.step(x=x,
                    y=y,
                    where='post',
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=label)
        else:
            plt.plot(x,
                    y,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=label)
    
    def plot_P_zk__vi_vf(self, vi, vf, Zk_name, max_one=False, sum_one=False, color=None, linestyle='-', linewidth=1.5, label=None, alpha=None, step=True):
        """
        plots a given :math:`P_k(\hat{z}|v_i,v_f)`.
        
    
        """
        # for colors, see https://matplotlib.org/3.1.0/gallery/color/named_colors.html
        
        P_zk__vi_vf = self.P_zk__vi_vf.loc[(self.P_zk__vi_vf.vi == vi) &
                                    (self.P_zk__vi_vf.vf == vf) &
                                    (self.P_zk__vi_vf.Zk_name == Zk_name)]
        
        
        y = P_zk__vi_vf.P_zk__vi_vf.values
        y = y[1:-1]
        
        if type(alpha) == pd.DataFrame:
            x = alpha.loc[(alpha.vi == vi) &
                          (alpha.Zk_name == Zk_name)].alpha.values
            x = x[0:-1]
        else:
            x = P_zk__vi_vf.q.values
            x = x[1:-1]
            
        if sum_one:
            y = y/np.sum(y)
            
        if max_one:
            y = y/np.max(y)
        
        if step:
            plt.step(x=x,
                    y=y,
                    where='post',
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=label)
        else:
            plt.plot(x,
                    y,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=label)

    

def _compute_P_vf__vi_z_with_bayes(P_vf__vi, P_z__vi, P_z__vi_vf, keep_P=False):
    
    col_vi = [('v','i')]
    
    cols_z = P_z__vi_vf[['v','z']].columns.to_list()
    
    P_vf__vi_z = P_z__vi_vf.merge(P_z__vi, how='left', on=col_vi+cols_z)
    
    P_vf__vi_z = P_vf__vi_z.merge(P_vf__vi.astype(float), how='left', on=col_vi)
        
    for vf in P_z__vi_vf.P_z__vi_vf.columns.to_list():
        idx = P_vf__vi_z.loc[P_vf__vi_z.P_z__vi > 0].index.values
        P_vf__vi_z.loc[idx, ('P_vf__vi_z', vf)] = P_vf__vi_z.loc[idx, ('P_vf__vi', vf)] * P_vf__vi_z.loc[idx, ('P_z__vi_vf', vf)] / P_vf__vi_z.loc[idx, ('P_z__vi', '')]
    
    if not keep_P:
        P_vf__vi_z.drop(['P_vf__vi','P_z__vi_vf', 'P_z__vi'], axis=1, level=0, inplace=True)
    
    return(P_vf__vi_z)

def _build_probability_maps(case, P, P_name = 'P_vf__vi_z', sound=1):
    # check P_vf__vi_z integrity :
    if P_name == 'P_vf__vi_z':
        if P.P_vf__vi_z.sum(axis=1).max() > 1:
            if sound >0:
                print('warning, max(sum_z(P_vf__vi_z))=',P.P_vf__vi_z.sum(axis=1).max())
        else:
            if sound > 1:
                print('check P_vf__vi_z ok')
    
    probability_maps = TransitionProbabilityLayers()
    
    J_with_P = case.discrete_J.reset_index().merge(right=P,
                                     how='left').set_index('index')
    
    # return(J_with_P)
    
    # print(J_with_P)
    for Ti in case.transitions.Ti.values():
        idx_vi = J_with_P.loc[J_with_P.v.i==Ti.vi].index.values
        for Tif in Ti.Tif.values():
            probability_map_data = np.zeros(case.map_i.data.shape)
            probability_map_data.flat[idx_vi] = J_with_P.loc[idx_vi, (P_name, Tif.vf)].values
            
            probability_maps.add_layer(Ti.vi, Tif.vf, probability_map_data)
        
    return(probability_maps)

