import pandas as pd

from .. import definition
from ..tools import np_suitable_integer_type
from .train_test_split import train_test_split_non_null_constraint
from ..allocation.scenario import compute_transition_probabilities

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm
from sklearn.model_selection import cross_val_score as sklearn_cross_val_score
from copy import deepcopy

class _Calibration():
    # def __init__(self):
        # self.a = None
    
 
    def fit(self, X, y):
        """

        """
        
        self.estimators = {}
        
        print('fitting...')
        for vi in X.keys():
            print('vi=',vi)
            
            self.estimators[vi] = self._new_estimator()
            self.estimators[vi].fit(X[vi], 
                                    y[vi])
            
    def predict(self, X, unit_measure=True):
        
        if X is pd.DataFrame:
            X = deepcopy(X)
            for vi in X.keys():
                X[vi] = X[vi].z.values
        
        y_predict = {}
        
        for vi in X.keys():
            y_predict[vi] = self.estimators[vi].predict(X[vi])
            
            y_predict[vi][y_predict[vi] < 0] = 0
            
            if unit_measure:
                s = y_predict[vi].sum(axis=0)
                id_column = s > 0
                
                y_predict[vi][:,id_column] = y_predict[vi][:,id_column] / s[id_column]
            
        return(y_predict)
    
    def predict_transition_probabilities(self,
                                         case,
                                         P_vf__vi,
                                         epsilon=0.05,
                                         n_iter_max=100,
                                         sound=0):
        if sound > 0:
            print('predict on unique z...')
        
        unique_Z = case.get_unique_z(output='pd')
        P_z__vi_vf = self.predict(unique_Z)
        
        tp = compute_transition_probabilities(case = case,
                                                unique_Z = unique_Z,
                                                P_z__vi_vf = P_z__vi_vf,
                                                P_vf__vi = P_vf__vi,
                                                epsilon=epsilon,
                                                n_iter_max=n_iter_max,
                                                sound=sound)
        
        
        return(tp)
    
    def predict_transition_probabilities_isl_exp(self,
                                                 case,
                                                 P_vf__vi,
                                                 id_J_exp,
                                                 epsilon=0.05,
                                                 n_iter_max=100,
                                                 sound=0):
        if sound > 0:
            print('predict on unique z...')
                
        tp = {}
        
        # isl
        unique_Z = case.get_unique_z(output='pd')
        P_z__vi_vf = self.predict(unique_Z)
                
        tp['isl'] = compute_transition_probabilities(case = case,
                                            unique_Z = unique_Z,
                                            P_z__vi_vf = P_z__vi_vf,
                                            P_vf__vi = P_vf__vi['isl'],
                                            epsilon=epsilon,
                                            n_iter_max=n_iter_max,
                                            sound=sound)
        
        # exp
        case_exp = case.copy()
        exp_condition = {}
        
        for vi in case.dict_vi_vf.keys():
            exp_condition[vi] = np.zeros(case.J[vi].shape).astype(bool)
            for id_vf, vf in enumerate(case.dict_vi_vf[vi]):
                exp_condition[vi] = exp_condition[vi] | id_J_exp[vi][vf]
            
            case_exp.keep_only(vi, exp_condition[vi], inplace=True)
        
        P_vf__vi_exp = {}
        for vi in case.dict_vi_vf.keys():
            P_vf__vi_exp[vi] = P_vf__vi['exp'][vi] * case.J[vi].size / case_exp.J[vi].size
        
        
        unique_Z_exp = case_exp.get_unique_z(output='pd')
        P_z__vi_vf_exp = self.predict(unique_Z_exp)
        
        for vi in case.dict_vi_vf.keys():
            for id_vf, vf in enumerate(case.dict_vi_vf[vi]):
                P_z__vi_vf_exp[vi][unique_Z_exp[vi][('z', 'distance_to_'+str(vf))] != 1, id_vf] = 0
            
            # unit measure
            s = P_z__vi_vf_exp[vi].sum(axis=0)
            id_column = s > 0
            P_z__vi_vf_exp[vi][:,id_column] = P_z__vi_vf_exp[vi][:,id_column] / s[id_column]
            
        tp_exp = compute_transition_probabilities(case = case_exp,
                                            unique_Z = unique_Z_exp,
                                            P_z__vi_vf = P_z__vi_vf_exp,
                                            P_vf__vi = P_vf__vi_exp,
                                            epsilon=epsilon,
                                            n_iter_max=n_iter_max,
                                            sound=sound)
        
        tp['exp'] = {}
        for vi in case.dict_vi_vf.keys():
            tp['exp'][vi] = np.zeros_like(tp['isl'][vi])
            tp['exp'][vi][exp_condition[vi], :] = tp_exp[vi]
        
        return(tp)
    
    def monte_carlo_score(self,
                          X,
                          y,
                          mc=10,
                          test_size=0.2,
                          return_all_scores=False,
                          split_method='nnc',
                          sound=0):
        
        # il faudrait se baser sur la fonction predict de l'objet qui vérifie la fermeture des probas...
        scores = {}
        
        estimator = self._new_estimator()
        
        if split_method == 'nnc':
            split_function = train_test_split_non_null_constraint
        elif split_method == 'normal':
            split_function = train_test_split
        
        for vi in X.keys():
            # if sound > 0:
                # print('vi='+str(vi))
            scores[vi] = []
            
            if sound > 1:
                forloop = tqdm(range(mc))
            else:
                forloop = range(mc)
            
            
            for imc in forloop:
                X_train, X_test, y_train, y_test = split_function(X[vi],
                                                                  y[vi],
                                                                  test_size=test_size)
                                                
                estimator.fit(X_train, y_train)
                
                scores[vi].append(estimator.score(X_test, y_test))
        
            if not return_all_scores:
                scores[vi] = np.mean(scores[vi])
                
        return(scores)
    
# def get_transition_probabilities_by_merging(Z, X, y, sound=0):
    
#     if sound > 0:
#         print('merge on all z...')
#     for vi in Z.keys():
#         if sound > 0:
#             print('\t vi='+str(vi))
#         col_names = pd.MultiIndex.from_tuples(Z[vi].columns.to_list())
#         Z_unique = pd.DataFrame(X[vi], columns=col_names)
#         for vf_idx in range(y[vi].shape[1]):    
#             Z_unique[('P', vf_idx)] = y[vi][:, vf_idx]
        
        
#         Z[vi] = Z[vi].merge(Z_unique, how='left').P.values
        
#     return(Z)

    # def feature_selection_by_score(self,
    #                                 case,
    #                                 method='cv',
    #                                 cv=5,
    #                                 mc=10,
    #                                 test_size=0.2,
    #                                 sound=0):
    #     """
    #     Feature selection by CrossValidation Score

    #     Parameters
    #     ----------
    #     case : TYPE
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     None.

    #     """
        
    #     results = {}
        
    #     for vi in case.J.keys():
            
    #         if sound>0:
    #             print('=!=!=!=!=!=')
    #             print('vi='+str(vi))
    #             print('=!=!=!=!=!=\n')
                
    #         case_select = case.select_vi(vi=vi, inplace=False)
            
    #         results[vi] = []
                        
    #         if method=='cv':
    #             to_exclude_score = self.cross_val_score(case_select,
    #                                                     cv=cv,
    #                                                     sound=sound)[vi]
    #         elif method == 'mc':
    #             to_exclude_score = self.monte_carlo_score(case_select,
    #                                                        mc=mc,
    #                                                        test_size=test_size,
    #                                                        sound=sound)[vi]
            
    #         while len(case_select.Z_names[vi])>1:
    #             if sound > 0:
    #                 print('remaining features :')
    #                 print(case_select.Z_names[vi])
    #                 print('')
                    
    #             results[vi].append({'z_names':case_select.Z_names[vi].copy(),
    #                                 'score':to_exclude_score})
                                
    #             to_exclude_name = None
    #             to_exclude_score = -1000
    #             for z_to_exclude in case_select.Z_names[vi]:
    #                 if sound > 0:
    #                     print('test '+z_to_exclude+' to exclude')
                    
    #                 if method=='cv':
    #                     score = self.cross_val_score(case_select.remove_z(vi, z_to_exclude, inplace=False),
    #                                                 cv=cv,
    #                                                 sound=sound)[vi]
    #                 elif method == 'mc':
    #                     score = self.monte_carlo_score(case_select,
    #                                                    mc=mc,
    #                                                    test_size=test_size,
    #                                                    split_method='nnc',
    #                                                    sound=sound)[vi]
                    
    #                 if sound > 0:
    #                     print('\t score : '+str(score))
                    
    #                 if score > to_exclude_score:
    #                     to_exclude_name = z_to_exclude
    #                     to_exclude_score = score
                
    #             case_select.remove_z(vi, to_exclude_name, inplace=True)
                
    #             if sound > 0:
    #                 print('')
    #                 print('============================')
    #                 print('')
                    
    #         results[vi].append({'z_names':case_select.Z_names[vi].copy(),
    #                             'score':to_exclude_score})
            
    #         if sound > 0:
    #             print('feature selection for vi='+str(vi)+ ':')
    #             print(case_select.Z_names)
        
    #     return(results)
    

# def compute_P_vf__vi_from_P_vf__vi_z(J, name='P_vf__vi'):
    
#     P_vf__vi = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('v','i')]))

#     P_vf__vi[('v','i')] = np.sort(J.v.i.unique()) 
    
#     for vi in np.sort(J.v.i.unique()):
#         for vf in J.P_vf__vi_z.columns.to_list():
#             P_vf__vi.loc[P_vf__vi.v.i==vi, (name, vf)] = J.loc[J.v.i==vi, ('P_vf__vi_z', vf)].sum() / J.loc[J.v.i==vi].index.size
            
#     return(P_vf__vi)     




def get_X_y(P, name='P_z__vi_vf'):
    X = {}
    y = {}
    
    for vi in P.keys():
        X[vi] = P[vi].z.values
        y[vi] = P[vi][name].values
        
    return(X, y)

# def compute_P_vi(J, name='P_vi'):
#     df = J.groupby([('v','i')]).size().reset_index(name=(name,''))
#     df[name] = df[name] / df[name].sum()
#     return(df)
#     # P_vi = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('v','i'), (name, '')]))
    
#     # for vi in J.v.i.unique():
#     #     P_vi.loc[P_vi.index.size] = [vi, J.loc[J.v.i.unique()]]

def compute_P_vf__vi(case):       
    P_vf__vi = {}
    
    for vi in case.dict_vi_vf.keys():
        P_vf__vi[vi] = []
        for vf in case.dict_vi_vf[vi]:
            P_vf__vi[vi].append((case.vf[vi] == vf).mean())
        
        P_vf__vi[vi] = np.array(P_vf__vi[vi])
    
    return(P_vf__vi)

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

def compute_P_z__vi_vf(case, name='P_z__vi_vf', n_smooth = None):
    P_z__vi_vf = compute_N_z_vi_vf(case, name=name)
    
    N_z_vi = compute_N_z_vi(case)
    
    for vi in P_z__vi_vf.keys():
        N_z_vi[vi].drop(['N_z_vi'], axis=1, level=0, inplace=True)
        
        for vf in P_z__vi_vf[vi].keys():
            P_z__vi_vf[vi][vf][(name, vf)] /= P_z__vi_vf[vi][vf][(name, vf)].sum()
            
            N_z_vi[vi] = N_z_vi[vi].merge(right=P_z__vi_vf[vi][vf],
                                          how='left')
            
        N_z_vi[vi].fillna(0, inplace=True)
            
        if n_smooth is not None:
            knr = KNeighborsRegressor(n_neighbors=n_smooth, weights=_distance_to_weights)
            knr.fit(N_z_vi[vi].z.values, N_z_vi[vi][name].values)
            
            N_z_vi[vi][[name]] = knr.predict(N_z_vi[vi].z.values)
            
            # scaling to have the sum equal to 1
            N_z_vi[vi][[name]] = N_z_vi[vi][name].values / N_z_vi[vi][name].sum(axis=0).values
    
    return(N_z_vi)
    
    
def compute_P_vf__vi_z(case, name='P_vf__vi_z', n_smooth=None):
    N_z_vi_vf = compute_N_z_vi_vf(case, name)
    P_vf__vi_z = compute_N_z_vi(case)
    
    for vi in N_z_vi_vf.keys():
        dict_columns_fillna = {}
        for vf in N_z_vi_vf[vi].keys():
            P_vf__vi_z[vi] = P_vf__vi_z[vi].merge(right=N_z_vi_vf[vi][vf],
                                                  how='left')
            dict_columns_fillna[(name,vf)] = 0
        
        # fillna
        P_vf__vi_z[vi].fillna(dict_columns_fillna, inplace=True)
        
        # divide
        P_vf__vi_z[vi][[name]] = P_vf__vi_z[vi][[name]].div(P_vf__vi_z[vi]['N_z_vi'],
                                                                            axis=0).values
    
        # remove N_z_vi column
        P_vf__vi_z[vi].drop(['N_z_vi'], axis=1, level=0, inplace=True)
        
        if n_smooth is not None:
            
            knr = KNeighborsRegressor(n_neighbors=n_smooth, weights=_distance_to_weights)
            knr.fit(P_vf__vi_z[vi].z.values, P_vf__vi_z[vi][name].values)
            
            P_vf__vi_z[vi][[name]] = knr.predict(P_vf__vi_z[vi].z.values)
            
            # il faudrait rescaler pour obtenir le même P_vf__vi
    
    return(P_vf__vi_z)

# def compute_P_vf__vi_z_with_bayes(P_vf__vi, P_z__vi, P_z__vi_vf, keep_P=False):
#     P_vf__vi_z = {}
    
#     for vi in P_z__vi_vf.keys():
#         P_vf__vi_z[vi] = P_z__vi_vf[vi].merge(P_z__vi[vi], how='left')
        
#         for vf in P_z__vi_vf[vi].P_z__vi_vf.columns.to_list():
#             P_vf__vi_z[vi][('P_vf__vi_z', vf)] = P_vf__vi[vi][vf] * P_vf__vi_z[vi][('P_z__vi_vf', vf)] / P_vf__vi_z[vi][('P_z__vi', '')]
        
#         if not keep_P:
#             P_vf__vi_z[vi].drop(['P_z__vi_vf', 'P_z__vi'], axis=1, level=0, inplace=True)
        
#         # P_vf__vi_z[cols_z] = P_vf__vi_z[cols_z].replace(to_replace=-1, value=np.nan).values
        
#     return(P_vf__vi_z)