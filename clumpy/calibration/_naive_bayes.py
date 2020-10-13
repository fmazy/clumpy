from ._calibration import _Calibration
from ..definition._case import Case
from ..definition._layer import TransitionProbabilityLayers
from ._calibration import compute_P_vf__vi, compute_P_vf__vi_z, compute_P_z__vi, compute_P_z__vi_vf

import sklearn.metrics
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class NaiveBayes(_Calibration):
    """
    Naive Bayes calibration.
    """
        
    def fit(self, J):
        """
        Fit the model using J as training data

        Parameters
        ----------
        J : pandas dataframe.
            A two level ``z`` (X) and ``P_vf__vi_z`` (y) columns are expected.

        Notes
        -----
        
        New attributes are availables :
            
            ``self.P_zk__vi``
            
            ``self.P_zk__vi_vf``
            
            ``self.P_vf__vi``

        """
        
        J = J.reindex(sorted(J.columns), axis=1)
        
        self._compute_P_zk__vi(J)
        self._compute_P_zk__vi_vf(J)
        self.P_vf__vi = compute_P_vf__vi(J)
        
    def predict(self, J):
        """
        Predict the target for the provided data.

        Parameters
        ----------
        J : pandas dataframe.
            A two level ``z`` feature column is expected.

        Returns
        -------
        J_predicted : pandas dataframe
            Target values above the ``P_vf__vi_z`` column.

        """
        J = J.reindex(sorted(J.columns), axis=1)
        
        col_vi = [('v','i')]
        cols_z = J[['z']].columns.to_list()
        
        J = J[col_vi+cols_z]
        
        P_z__vi = self._compute_naive_P_z__vi(J)
        
        P_z__vi_vf = self._compute_naive_P_z__vi_vf(J)
        
        P_vf__vi_z = compute_P_vf__vi_z_with_bayes(self.P_vf__vi, P_z__vi, P_z__vi_vf)
        
        J = J.reset_index(drop=False).merge(P_vf__vi_z, how='left').set_index('index')
        
        for vi in J.v.i.unique():
            J.loc[J.v.i == vi, ('P_vf__vi_z', vi)] = 1 - J.loc[J.v.i == vi].P_vf__vi_z.sum(axis=1)
        
        J['P_vf__vi_z'] = J['P_vf__vi_z'].fillna(0)
        
        J = J.reindex(sorted(J.columns), axis=1)
        
        return(J)
    
    def score(self, J, y):
        """
        Return the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(). The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0. It returns the coefficient R^2 for each initial states.

        Parameters
        ----------
        J : pandas dataframe.
            A two level ``z`` feature column is expected.
        y : numpy array.
            True ``P_vf__vi_z`` values for J in the same order.

        Returns
        -------
        score : list of floats
            R^2 of self.predict(J) wrt. y for each ``vi`` in the ascending order.

        """
        J = J.reindex(sorted(J.columns), axis=1)
        
        J_predict = self.predict(J)
        J_predict.reset_index(inplace=True, drop=True)
        
        s = []
        for vi in J_predict.v.i.unique():
            id_vi_pixels = J_predict.loc[J_predict.v.i == vi].index.values
            
            P_vf__vi_z = J_predict.loc[id_vi_pixels, 'P_vf__vi_z'].values
            
            # focus on different final state
            list_vf = J_predict.P_vf__vi_z.columns.to_list()
            id_vf_columns_without_vi = list(np.arange(len(list_vf)))
            id_vf_columns_without_vi.remove(list_vf.index(vi))
            
            y_true = y[id_vi_pixels,:]
            y_true = y_true[:, id_vf_columns_without_vi]
            
            # print(y_true)
            
            y_predict = P_vf__vi_z[:,id_vf_columns_without_vi]
            
            # print(y_predict)
            
            s.append(sklearn.metrics.r2_score(y_true, y_predict))
        
        return(s)
        
    # def transition_probability_maps(self, case:Case, P_vf__vi = None):
    #     """
    #     Infer transition probability maps 

    #     Parameters
    #     ----------
    #     case : Case
    #         The case which have to be discretized..
    #     P_vf__vi : Pandas DataFrame (default=None)
    #         The transition matrix. If ``None``, the self computed one is used.
        
    #     Returns
    #     -------
    #     maps : TransitionProbabilityLayers
    #         The transition probability maps
    #     """
    #     if type(P_vf__vi) == type(None):
    #         self._compute_P_vf__vi(case)
    #         P_vf__vi = self.P_vf__vi
        
    #     P_z__vi = self._compute_naive_P_z__vi(case).fillna(0)
    #     P_z__vi_vf = self._compute_naive_P_z__vi_vf(case).fillna(0)
        
    #     P_vf__vi_z = _compute_P_vf__vi_z_with_bayes(P_vf__vi, P_z__vi, P_z__vi_vf).fillna(0)
        
    #     # return(P_vf__vi_z)
        
    #     # print(P_vf__vi_z)
        
    #     maps = _build_probability_maps(case, P_vf__vi_z)
        
    #     return(maps)
        
    def _compute_P_zk__vi(self, J):
        self.P_zk__vi = {}
        
        for vi in J.v.i.unique():
            J_vi = J.loc[(J.v.i==vi)].copy()
            
            for zk_name in J.z.columns.to_list():
                # check if the feature is equal to NaN
                if J_vi[('z', zk_name)].isna().sum() != J_vi.index.size:
                    self.P_zk__vi[(vi, zk_name)] = compute_P_z__vi(J_vi[[('v','i'), ('z', zk_name)]],
                                                                   name=('P_zk__vi', zk_name))
                        
    def _compute_P_zk__vi_vf(self, J):       
        
        self.P_zk__vi_vf = {}
        
        for vi in J.v.i.unique():
            J_vi = J.loc[(J.v.i==vi)]
            for vf in J_vi.v.f.unique():
                if vf != vi:
                    J_vi_vf = J_vi.loc[(J.v.f==vf)]
                    for zk_name in J.z.columns.to_list():
                        # check if the feature is equal to NaN
                        if J_vi_vf[('z', zk_name)].isna().sum() != J_vi_vf.index.size:
                            self.P_zk__vi_vf[(vi, vf, zk_name)] = compute_P_z__vi_vf(J_vi_vf[[('v','i'), ('v','f'), ('z', zk_name)]],
                                                                                     name='P_z'+zk_name+'__vi_vf')

    
    def _compute_naive_P_z__vi(self, J):        
        col_vi = [('v', 'i')]
        cols_z = J[['z']].columns.to_list()
        
        P_z__vi = J[col_vi + cols_z].drop_duplicates()
        
        for vi in P_z__vi.v.i.unique():
            for (vix, zk_name), P_zk__vi in self.P_zk__vi.items():
                if vix == vi:
                    P_z__vi = P_z__vi.merge(P_zk__vi, how='left')
            
            P_z__vi.loc[P_z__vi.v.i==vi,('P_z__vi','')] = P_z__vi.loc[P_z__vi.v.i==vi].P_zk__vi.product(axis=1, min_count=1)
            
            P_z__vi.drop('P_zk__vi', axis=1, level=0, inplace=True)
        
        return(P_z__vi)
    
    def _compute_naive_P_z__vi_vf(self, J):
        col_vi = [('v', 'i')]
        cols_z = J[['z']].columns.to_list()
        
        P_z__vi_vf = J[col_vi + cols_z].drop_duplicates()
        
        for vi in P_z__vi_vf.v.i.unique():
            list_vf_according_to_vi = []
            P_zk__vi_vf_names_according_to_vf = {}
            for (vix, vf, zk_name), P_zk__vi_vf in self.P_zk__vi_vf.items():
                if vix == vi:
                    P_z__vi_vf = P_z__vi_vf.merge(P_zk__vi_vf, how='left')
                    
                    if vf not in list_vf_according_to_vi:
                        list_vf_according_to_vi.append(vf)
                        P_zk__vi_vf_names_according_to_vf[vf] = []
                    
                    if ('P_z'+zk_name+'__vi_vf', vf) not in P_zk__vi_vf_names_according_to_vf[vf]:
                        P_zk__vi_vf_names_according_to_vf[vf].append(('P_z'+zk_name+'__vi_vf', vf))
            
            
            for vf in list_vf_according_to_vi:
                P_z__vi_vf.loc[P_z__vi_vf.v.i==vi,('P_z__vi_vf',vf)] = P_z__vi_vf.loc[P_z__vi_vf.v.i==vi, P_zk__vi_vf_names_according_to_vf[vf]].product(axis=1, min_count=1)
                            
                P_z__vi_vf.drop(P_zk__vi_vf_names_according_to_vf[vf], axis=1, inplace=True)
                
        return(P_z__vi_vf)
    
    def plot_P_zk__vi(self, vi, Zk_name, max_one=False, sum_one=False, color=None, linestyle='-', linewidth=1.5, label=None, alpha=None, step=True):
        """
        plots a given :math:`P_k(\hat{z}|v_i,v_f)`.
        
        """
        # for colors, see https://matplotlib.org/3.1.0/gallery/color/named_colors.html
        
        P_zk__vi = self.P_zk__vi[(vi, Zk_name)]
        print(P_zk__vi)
        plt.scatter(P_zk__vi.z[Zk_name].values, P_zk__vi.P_zk__vi[Zk_name].values)
        
        # P_zk__vi = self.P_zk__vi.loc[(self.P_zk__vi.vi == vi) &
        #                       (self.P_zk__vi.Zk_name == Zk_name)]
        
        # y = P_zk__vi.P_zk__vi.values
        # y = y[1:-1]
        
        # if type(alpha) == pd.DataFrame:
        #     x = alpha.loc[(alpha.vi == vi) &
        #                   (alpha.Zk_name == Zk_name)].alpha.values
        #     x = x[0:-1]
        # else:
        #     x = P_zk__vi.q.values
        #     x = x[1:-1]
        
        # if sum_one:
        #     y = y/np.sum(y)
            
        # if max_one:
        #     y = y/np.max(y)
        
        # if step:
        #     plt.step(x=x,
        #             y=y,
        #             where='post',
        #             color=color,
        #             linestyle=linestyle,
        #             linewidth=linewidth,
        #             label=label)
        # else:
        #     plt.plot(x,
        #             y,
        #             color=color,
        #             linestyle=linestyle,
        #             linewidth=linewidth,
        #             label=label)
    
    def plot_P_zk__vi_vf(self, vi, vf, Zk_name, max_one=False, sum_one=False, color=None, linestyle='-', linewidth=1.5, label=None, alpha=None, step=True):
        """
        plots a given :math:`P_k(\hat{z}|v_i,v_f)`.
        
    
        """
        
        P_zk__vi_vf = self.P_zk__vi_vf[(vi, vf, Zk_name)]
        print(P_zk__vi_vf)
        plt.scatter(P_zk__vi_vf.z[Zk_name].values, P_zk__vi_vf[('P_z'+Zk_name+'__vi_vf', vf)].values)
        
        # for colors, see https://matplotlib.org/3.1.0/gallery/color/named_colors.html
        
        # P_zk__vi_vf = self.P_zk__vi_vf.loc[(self.P_zk__vi_vf.vi == vi) &
        #                             (self.P_zk__vi_vf.vf == vf) &
        #                             (self.P_zk__vi_vf.Zk_name == Zk_name)]
        
        
        # y = P_zk__vi_vf.P_zk__vi_vf.values
        # y = y[1:-1]
        
        # if type(alpha) == pd.DataFrame:
        #     x = alpha.loc[(alpha.vi == vi) &
        #                   (alpha.Zk_name == Zk_name)].alpha.values
        #     x = x[0:-1]
        # else:
        #     x = P_zk__vi_vf.q.values
        #     x = x[1:-1]
            
        # if sum_one:
        #     y = y/np.sum(y)
            
        # if max_one:
        #     y = y/np.max(y)
        
        # if step:
        #     plt.step(x=x,
        #             y=y,
        #             where='post',
        #             color=color,
        #             linestyle=linestyle,
        #             linewidth=linewidth,
        #             label=label)
        # else:
        #     plt.plot(x,
        #             y,
        #             color=color,
        #             linestyle=linestyle,
        #             linewidth=linewidth,
        #             label=label)

    



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

