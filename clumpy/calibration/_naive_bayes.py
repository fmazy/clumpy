from ._calibration import _Calibration
from ..definition._case import Case
from ..definition._layer import TransitionProbabilityLayers

import sklearn.metrics
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class NaiveBayes(_Calibration):
    """
    Naive Bayes calibration method.
    """
    # def __init__(self):
        # super().__init__()
        
    def fit(self, J):
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
        
        self._compute_P_zk__vi(J)
        self._compute_P_zk__vi_vf(J)
        self.P_vf__vi = self._compute_P_vf__vi(J)
        
    def predict(self, J):
        
        J = J.copy()
        col_vi = [('v','i')]
        cols_z = J[['z']].columns.to_list()
        
        J = J[col_vi+cols_z]
        
        P_z__vi = self._compute_naive_P_z__vi(J)
        P_z__vi_vf = self._compute_naive_P_z__vi_vf(J)
        print(P_z__vi_vf)
        P_vf__vi_z = _compute_P_vf__vi_z_with_bayes(self.P_vf__vi, P_z__vi, P_z__vi_vf)
        print(P_vf__vi_z)
        J = J.reset_index(drop=False).merge(P_vf__vi_z, how='left').set_index('index')
        
        for vi in J.v.i.unique():
            J.loc[J.v.i == vi, ('P_vf__vi_z', vi)] = 1 - J.loc[J.v.i == vi].P_vf__vi_z.sum(axis=1)
        
        J = J.reindex(sorted(J.columns), axis=1)
        
        return(J)
    
    def score(self, J, y):
        
        J = J.copy()
        
        J_predict = self.predict(J)
        J_predict.reset_index(inplace=True, drop=True)
        
        s = []
        for vi in J_predict.v.i.unique():
            idx = J_predict.loc[J_predict.v.i == vi].index.values
            
            P_vf__vi_z = J_predict.loc[idx, 'P_vf__vi_z'].values
            
            # focus on different final state
            list_vf = J_predict.P_vf__vi_z.columns.to_list()
            idx_vi = list_vf.index(vi)
            idx_vf = list(np.arange(len(list_vf)))
            idx_vf.remove(idx_vi)
            
            y_true = y[idx,:]
            y_true = y_true[:, idx_vf]
            
            y_predict = P_vf__vi_z[:,idx_vf]
            
            s.append(sklearn.metrics.r2_score(y_true, y_predict))
        
        return(s)
        
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
        self.P_zk__vi = {}
        
        for vi in J.v.i.unique():
            J_vi = J.loc[(J.v.i==vi)]
            for zk_name in J.z.columns.to_list():
                self.P_zk__vi[(vi, zk_name)] = self.compute_P_z__vi(J_vi[[('v','i'), ('z', zk_name)]], name=('P_zk__vi', zk_name), output='return')
                        
    def _compute_P_zk__vi_vf(self, J):       
        
        self.P_zk__vi_vf = {}
        
        for vi in J.v.i.unique():
            J_vi = J.loc[(J.v.i==vi)]
            for vf in J_vi.v.f.unique():
                if vf != vi:
                    J_vi_vf = J_vi.loc[(J.v.f==vf)]
                    for zk_name in J.z.columns.to_list():
                        self.P_zk__vi_vf[(vi, vf, zk_name)] = self.compute_P_z__vi_vf(J_vi_vf[[('v','i'), ('v','f'), ('z', zk_name)]], name='P_z'+zk_name+'__vi_vf', output='return')

    
    def _compute_naive_P_z__vi(self, J):        
        col_vi = [('v', 'i')]
        cols_z = J[['z']].columns.to_list()
        
        P_z__vi = J[col_vi + cols_z].drop_duplicates()
        
        for (vi, zk_name), P_zk__vi in self.P_zk__vi.items():
            P_z__vi = P_z__vi.merge(P_zk__vi, how='left')
        
        P_z__vi[('P_z__vi','')] = P_z__vi.P_zk__vi.product(axis=1, min_count=1)
        
        P_z__vi.drop('P_zk__vi', axis=1, level=0, inplace=True)
        
        return(P_z__vi)
    
    def _compute_naive_P_z__vi_vf(self, J):
        col_vi = [('v', 'i')]
        cols_z = J[['z']].columns.to_list()
        
        P_z__vi_vf = J[col_vi + cols_z].drop_duplicates()
        
        list_vf = []
        list_zk_name = []
        for (vi, vf, zk_name), P_zk__vi_vf in self.P_zk__vi_vf.items():
            list_vf.append(vf)
            list_zk_name.append(zk_name)
            P_z__vi_vf = P_z__vi_vf.merge(P_zk__vi_vf, how='left')
        
        for vf in list_vf:
            P_z__vi_vf[('P_z__vi_vf',vf)] = P_z__vi_vf[[('P_z'+zk_name+'__vi_vf', vf) for zk_name in list_zk_name]].product(axis=1, min_count=1)
        
        P_z__vi_vf.drop(['P_z'+zk_name+'__vi_vf' for zk_name in list_zk_name], level=0, axis=1, inplace=True)
        
        return(P_z__vi_vf)
        # P_z__vi[('P_z__vi','')] = P_z__vi.P_zk__vi.product(axis=1, min_count=1)
        
        # P_z__vi.drop('P_zk__vi', axis=1, level=0, inplace=True)
        
        # return(P_z__vi)
        
        
        # J = J.copy()
        
        # if ('v','f') in J.columns:
        #     J = J.drop(('v', 'f'), axis=1)
        
        # P_z__vi_vf = J.drop_duplicates()
        
        # T = list(self.P_zk__vi_vf.groupby(['vi','vf']).size().index)
        # Z = list(self.P_zk__vi_vf.groupby(['vi','Zk_name']).size().index)
        
        # print(T)
        # print(Z)
        
        # for Tk in T:
        #     vi = Tk[0]
        #     vf = Tk[1]
        #     for Zk in Z:
        #         if Zk[0] == vi:
        #             Zk_name = Zk[1]
        #             cols = [('v', 'i'), ('z', Zk_name), ('P_zk__vi_vf', Zk_name)]
        #             cols = pd.MultiIndex.from_tuples(cols)
        #             df = pd.DataFrame(columns=cols)
        #             df[cols] = self.P_zk__vi_vf.loc[(self.P_zk__vi_vf.vi == vi) &
        #                                              (self.P_zk__vi_vf.vf == vf) &
        #                                              (self.P_zk__vi_vf.Zk_name == Zk_name), ['vi', 'q', 'P_zk__vi_vf']]
                    
        #             P_z__vi_vf = P_z__vi_vf.merge(df.astype(float), how='left')
            
        #     P_z__vi_vf.loc[P_z__vi_vf.v.i==vi, ('P_z__vi_vf', vf)] = P_z__vi_vf.loc[P_z__vi_vf.v.i==vi].P_zk__vi_vf.product(axis=1, min_count=1)
            
        #     P_z__vi_vf.drop('P_zk__vi_vf', axis=1, level=0, inplace=True)
                
        # return(P_z__vi_vf)
    
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
            
    P_vf__vi_z = P_z__vi_vf.merge(P_z__vi, how='left')
    
    P_vf__vi_z = P_vf__vi_z.merge(P_vf__vi, how='left')
        
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

