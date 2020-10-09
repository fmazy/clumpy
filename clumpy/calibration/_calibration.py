import pandas as pd

from .. import definition
from ..tools import np_suitable_integer_type
from .train_test_split import train_test_split_non_null_constraint

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm
from sklearn.model_selection import cross_val_score as sklearn_cross_val_score

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
            
    def predict(self, X):
        y_predict = {}
        
        for vi in X.keys():
            y_predict[vi] = self.estimators[vi].predict(X[vi])
                    
        return(y_predict)
    
    def predict_on_case(self, case, sound=0):
        if sound > 0:
            print('predict on unique z...')
        X = case.get_unique_z(output='np')
        
        y_predict = self.predict(X)
        
        Z = case.get_z_as_dataframe()
        if sound > 0:
            print('merge on all z...')
        for vi in Z.keys():
            if sound > 0:
                print('\t vi='+str(vi))
            col_names = pd.MultiIndex.from_tuples(Z[vi].columns.to_list())
            Z_unique = pd.DataFrame(X[vi], columns=col_names)
            for vf_idx in range(y_predict[vi].shape[1]):    
                Z_unique[('P', vf_idx)] = y_predict[vi][:, vf_idx]
            
            Z[vi] = Z[vi].merge(Z_unique, how='left')
            
            Z[vi] = Z[vi].P.values
            
        return(Z)
    
    def monte_carlo_score(self,
                          X,
                          y,
                          mc=10,
                          test_size=0.2,
                          return_all_scores=False,
                          split_method='nnc',
                          sound=0):
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


# def get_features_and_targets(P_vf__vi_z):
#     """
#     Get features and targets of a transition probabilities dataframe.

#     Parameters
#     ----------
#     P_vf__vi_z : pandas dataframe 
#         Transition probabilities. 
    
#     Returns
#     -------
#     ``X`` the initial state and the features and ``Y`` the targets, i.e. the transitions probabilities
#     """    
#     col_vi = [('v', 'i')]
#     cols_z = P_vf__vi_z[['z']].columns.to_list()
#     cols_P = P_vf__vi_z[['P_vf__vi_z']].columns.to_list()
    
#     X = P_vf__vi_z[col_vi+cols_z]
#     Y =  P_vf__vi_z[cols_P]
    
#     return(X,Y)
    
#     # def transition_probability_maps():

def get_X_y(P, name='P_vf__vi_z'):
    X = {}
    y = {}
    
    for vi in P.keys():
        X[vi] = P[vi].z.values
        y[vi] = P[vi][name].values
        
    return(X, y)

def compute_P_vi(J, name='P_vi'):
    df = J.groupby([('v','i')]).size().reset_index(name=(name,''))
    df[name] = df[name] / df[name].sum()
    return(df)
    # P_vi = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('v','i'), (name, '')]))
    
    # for vi in J.v.i.unique():
    #     P_vi.loc[P_vi.index.size] = [vi, J.loc[J.v.i.unique()]]

def compute_P_vf__vi(case, name='P_vf__vi'):       
    P_vf__vi = {}
    
    for vi in case.Z.keys():
        P_vf__vi[vi] = {}
        df = pd.DataFrame(case.vf[vi], columns=['vf'])
        df = df.groupby(by='vf').size().reset_index(name='N')
        
        vf = df.vf.values
        N = df.N.values
        N = N / N.sum()
        
        for i in range(len(vf)):
            if vf[i] != vi:
                P_vf__vi[vi][vf[i]] = N[i]
    
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

# def compute_P_vf__vi_from_P_vf__vi_z(P_vf__vi)



def compute_P_vf__vi_z_with_bayes(P_vf__vi, P_z__vi, P_z__vi_vf, keep_P=False):
    P_vf__vi_z = {}
    
    for vi in P_z__vi_vf.keys():
        P_vf__vi_z[vi] = P_z__vi_vf[vi].merge(P_z__vi[vi], how='left')
        
        for vf in P_z__vi_vf[vi].P_z__vi_vf.columns.to_list():
            P_vf__vi_z[vi][('P_vf__vi_z', vf)] = P_vf__vi[vi][vf] * P_vf__vi_z[vi][('P_z__vi_vf', vf)] / P_vf__vi_z[vi][('P_z__vi', '')]
        
        if not keep_P:
            P_vf__vi_z[vi].drop(['P_z__vi_vf', 'P_z__vi'], axis=1, level=0, inplace=True)
        
        # P_vf__vi_z[cols_z] = P_vf__vi_z[cols_z].replace(to_replace=-1, value=np.nan).values
        
    return(P_vf__vi_z)