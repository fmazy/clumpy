# -*- coding: utf-8 -*-

import numpy as np

from ..kde import GKDE
from . import patches
from ..scenario import TransitionMatrix
import pandas as pd

class Calibrator():
    def __init__(self,
                 case,
                 p_min=0.5 * 10**(-4),
                 n_min=500,
                 n_kde_max=10**4,
                 n_predict_max=2*10**4,
                 n_jobs=1,
                 verbose=0):
        self.case = case
        self.p_min = p_min
        self.n_min = n_min
        self.n_kde_max = n_kde_max
        self.n_predict_max = n_predict_max
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self,
            initial_luc_layer,
            final_luc_layer,
            start_luc_layer,
            region_train,
            region_eval):

        self.start_luc_layer = start_luc_layer
        
        self.region_eval = region_eval
        
        self._make_case(initial_luc_layer,
                        final_luc_layer,
                        start_luc_layer,
                        region_train,
                        region_eval)

        self._get_low_high_bounded_features()

        self._fit_P_x__u_v()

        self._fit_P_y__u()

        return(self)

    def fit_patches(self,
                    initial_luc_layer,
                    final_luc_layer,
                    isl_exp=False,
                    neighbors_structure='queen'):

        if isl_exp:
            self._patches = patches.analyse_isl_exp(self.case,
                                                    initial_luc_layer,
                                                    final_luc_layer,
                                                    neighbors_structure=neighbors_structure)

            self._patches_isl_ratio = patches.compute_isl_ratio(self._patches)

        else:
            self._patches = patches.analyse(self.case,
                                            initial_luc_layer,
                                            final_luc_layer,
                                            neighbors_structure=neighbors_structure)

        return(self._patches)

    def transition_matrix(self):

        list_u = np.sort(list(self.case.params.keys()))
        # list_v__u = [self.case.params[u]['v'] for u in list_u]

        list_v = []

        unique_v__u = {}
        proba_v__u = {}

        for u in list_u:
            unique_v__u[u], proba_v__u[u] = np.unique(
                self._v_u[u], return_counts=True)
            proba_v__u[u] = proba_v__u[u] / proba_v__u[u].sum()
            # proba_v__u[u] /= np.sum(proba_v__u[u])

            for v in unique_v__u[u]:
                if v not in list_v:
                    list_v.append(v)

        list_v = list(np.sort(list_v).astype(int))

        M = np.zeros((len(list_u), len(list_v)))
        M.fill(np.nan)

        for id_u, u in enumerate(list_u):
            for id_v, v in enumerate(list_v):
                if v in unique_v__u[u]:
                    M[id_u, id_v] = proba_v__u[u][list(
                        unique_v__u[u]).index(v)]

        return(TransitionMatrix(M, list_u, list_v))
    
    def export_marginals(self, path_prefix, n=300):
        for u in self._calibrated_transitions_u.keys():
            for k, feature in enumerate(self.case.params[u]['features']):
                
                if feature[0] == 'layer':
                    name = feature[1].name
                elif feature[0] == 'distance':
                    name = 'distance_to_'+str(feature[1])
                
                x = np.linspace(self._X_u[u][:,k].min(), self._X_u[u][:,k].max(), n)
                
                df = pd.DataFrame(x, columns=[name])
                
                gkde = self._gkde_P_y__u[u]
                df['P(x|u'+str(u)+')'] = gkde.marginal(x, k)
                
                for id_v, v in enumerate(self._calibrated_transitions_u[u]):
                    
                    gkde = self._gkde_P_x__u_v[(u, v)]
                    df['P(x|u'+str(u)+',v'+str(v)+')'] = gkde.marginal(x, k)
                
                df.to_csv(path_prefix+'_u'+str(u)+'_'+name+'.csv', index=False)
                

    def _make_case(self,
                   initial_luc_layer,
                   final_luc_layer,
                   start_luc_layer,
                   region_train=None,
                   region_eval=None):
        if self.verbose > 0:
            print('Make case...')

        self._J_X_u, self._X_u, self._v_u = self.case.make(initial_luc_layer,
                                                           final_luc_layer,
                                                           region=region_train)
        self._J_Y_u, self._Y_u = self.case.make(start_luc_layer,
                                                region=region_eval)

        if self.verbose > 0:
            print('done')

    def _get_low_high_bounded_features(self):
        self._low_bounded_features_u = {}
        self._high_bounded_features_u = {}

        for u, params in self.case.params.items():
            low_bounded_features = []
            high_bounded_features = []

            for k, (feature_type, info) in enumerate(params['features']):
                if feature_type == 'distance':
                    low_bounded_features.append(k)
                elif feature_type == 'layer':
                    if info.low_bounded:
                        low_bounded_features.append(k)
                    if info.high_bounded:
                        high_bounded_features.append(k)

            self._low_bounded_features_u[u] = low_bounded_features
            self._high_bounded_features_u[u] = high_bounded_features

    def _fit_P_x__u_v(self, X_u=None, v_u=None):

        if X_u is None:
            X_u = self._X_u
        if v_u is None:
            v_u = self._v_u

        if self.verbose > 0:
            print('estimating P(y|u,v)')

        self._calibrated_transitions_u = {}
        self._gkde_P_x__u_v = {}

        for u in X_u.keys():

            self._calibrated_transitions_u[u] = []

            for idv, v in enumerate(self.case.params[u]['v']):
                if v != u:
                    if self.verbose > 0:
                        print('\tu='+str(u)+', v='+str(v))

                    X_u_v = X_u[u][v_u[u] == v]

                    if X_u_v.shape[0] >= self.n_min and \
                            X_u_v.shape[0] / v_u[u].shape[0] > self.p_min:

                        self._gkde_P_x__u_v[(u, v)] = GKDE(
                            h='scott',
                            low_bounded_features=self._low_bounded_features_u[u],
                            high_bounded_features=self._high_bounded_features_u[u],
                            n_predict_max=self.n_predict_max,
                            n_jobs=self.n_jobs,
                            verbose=self.verbose - 1)

                        if self.verbose > 0:
                            print('\tX_u_v shape : '+str(X_u_v.shape))
                            print('\tKDE fitting according to X_u_v')

                        self._gkde_P_x__u_v[(u, v)].fit(X_u_v)

                        if self.verbose > 0:
                            print('\th='+str(self._gkde_P_x__u_v[(u, v)]._h))

                        self._calibrated_transitions_u[u].append(v)

                    else:
                        if self.verbose > 0:
                            print(
                                '\tnot enough samples according to n_min and p_min.')

                if self.verbose > 0:
                    print('\t----------------')

        if self.verbose > 0:
            print('fitting GKDE P(x|u,v) done\n============================')

    def _estimate_P_Y__v(self, Y, u, log_return=False):

        P_Y__v = None

        list_v = []

        for idv, v in enumerate(self._calibrated_transitions_u[u]):
            if v != u:
                pdf = self._gkde_P_x__u_v[(u, v)].predict(Y)

                list_v.append(v)

                if P_Y__v is None:
                    P_Y__v = pdf[:, None]
                else:
                    P_Y__v = np.hstack((P_Y__v, pdf[:, None]))
        
        if log_return:
            return(_log(P_Y__v), list_v)
        else:
            return(P_Y__v, list_v)

    def _fit_P_y__u(self, Y_u=None):
        if Y_u is None:
            Y_u = self._Y_u

        if self.verbose > 0:
            print('estimating P(y|u)')

        # P_y__u = {}
        self._gkde_P_y__u = {}

        for u, Y in Y_u.items():
            if self.verbose > 0:
                print('\tu='+str(u))
                print('\tkde fit')

            if Y.shape[0] > self.n_kde_max:
                Y_train = Y[np.random.choice(Y.shape[0], self.n_kde_max)]
            else:
                Y_train = Y

            if self.verbose > 0:
                print('\tY_train shape : '+str(Y_train.shape))
                print('\tKDE fitting according to Y_train')

            self._gkde_P_y__u[u] = GKDE(h='scott',
                                        low_bounded_features=self._low_bounded_features_u[u],
                                        high_bounded_features=self._high_bounded_features_u[u],
                                        forbid_null_value=True,
                                        n_predict_max=self.n_predict_max,
                                        n_jobs=self.n_jobs,
                                        verbose=self.verbose - 1)
            print(Y_train.shape)
            self._gkde_P_y__u[u].fit(Y_train)

            if self.verbose > 0:
                print('\th='+str(self._gkde_P_y__u[u]._h))

            if self.verbose > 0:
                print('\t----------------')
        if self.verbose > 0:
            print('fitting GKDE P(y|u) done\n============================')

    def _estimate_P_Y(self, Y, u, log_return=False):
        P_Y = self._gkde_P_y__u[u].predict(Y)[:, None]
        
        if log_return:
            return(_log(P_Y))
        else:
            return(P_Y)


def _compute_P_v__Y(P_v, P_Y, P_Y__v, list_v, verbose=0):
    
    # log is better near 0 for divions
    log_P_v__Y = _log(P_Y__v) - _log(P_Y)
    log_P_v__Y += _log(P_v) - _log(np.exp(log_P_v__Y).mean(axis=0))
        
    s = np.exp(log_P_v__Y).sum(axis=1)

    if np.sum(s > 1) > 0:
        if verbose > 0:
            print('\tWarning, uncorrect probabilities have been detected.')
            print('\tSome global probabilities may be to high.')
            print('\tFor now, some corrections are made.')

        n_corrections_max = 100
        n_corrections = 0

        while np.sum(s > 1) > 0 and n_corrections < n_corrections_max:
            id_anomalies = s > 1

            log_P_v__Y[id_anomalies] = log_P_v__Y[id_anomalies] - \
                _log(s[id_anomalies])[:, None]
            
            log_P_v__Y += _log(P_v) - _log(np.exp(log_P_v__Y).mean(axis=0))
            
            n_corrections += 1
            s = np.sum(log_P_v__Y, axis=1)

        if verbose > 0:
            print('\tCorrections done in '+str(n_corrections)+' iterations.')

    # avoid nan values
    P_v__Y = np.nan_to_num(np.exp(log_P_v__Y))

    return(P_v__Y)

def _log(x):
    return(np.log(x,
                  out = np.zeros_like(x).fill(-np.inf),
                  where = x > 0))