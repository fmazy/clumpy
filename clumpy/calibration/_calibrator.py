# -*- coding: utf-8 -*-

import numpy as np

from ..kde import GKDE

class Calibrator():
    def __init__(self,
                 case,
                 p_min = 0.5 * 10**(-4),
                 n_min = 500,
                 n_kde_max = 10**4,
                 n_predict_max = 2*10**4,
                 n_jobs = 1,
                 verbose = 0):
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
            start_luc_layer):
        
        self._make_case(initial_luc_layer,
                        final_luc_layer,
                        start_luc_layer)
        
        self._get_low_high_bounded_features()
        
        self._estimate_P_y__u_v(self._X_u, self._v_u, self._Y_u)
        
        self._estimate_P_y__u(self._Y_u)
    
    def transition_matrix(self):
        
        list_u = np.sort(list(self.case.params.keys()))
        # list_v__u = [self.case.params[u]['v'] for u in list_u]
        
        list_v = []
        
        unique_v__u = {}
        proba_v__u = {}
        
        
        for u in list_u:
            unique_v__u[u], proba_v__u[u] = np.unique(self._v_u[u], return_counts=True)
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
                    M[id_u, id_v] = proba_v__u[u][list(unique_v__u[u]).index(v)]
        
        return(M, list_u, list_v)
    
    def transition_probabilities(self, M, list_u, list_v):
        P_v__u_y = {}
        list_v__u = {}
        
        for id_u, u in enumerate(list_u):
            if self.verbose > 0:
                print('u='+str(u))
            
            id_v_to_estimate = []
            for id_v, v in enumerate(list_v):
                if u != v and M[id_u, id_v] > 0:
                    if v not in self._calibrated_transitions_u[u]:
                        if self.verbose > 0:
                            print('\tWarning : transition '+str(u)+' -> '+str(v)+' has not been calibrated and is then ignored.')
                    else:
                        id_v_to_estimate.append(id_v)
            
            list_v__u[u] = list(np.array(list_v)[id_v_to_estimate])
            
            if self.verbose > 0:
                print('\tFirst P(y|u,v) computation')
            
            P_v__u = M[id_u, id_v_to_estimate]
            
            
            
            P_v__u_y[u] = self._P_y__u_v[u] / self._P_y__u[u]
            P_v__u_y[u] *= P_v__u / P_v__u_y[u].mean(axis=0)
            
            s = P_v__u_y[u].sum(axis=1)
                        
            if np.sum(s > 1) > 0:
                if self.verbose > 0:
                    print('\tWarning, uncorrect probabilities have been detected.')
                    print('\tSome global probabilities may be to high.')
                    print('\tFor now, some corrections are made.')
                
                n_corrections_max = 100
                n_corrections = 0
                
                while np.sum(s > 1) > 0 and n_corrections < n_corrections_max:
                    id_anomalies = s > 1
                    
                    P_v__u_y[u][id_anomalies] = P_v__u_y[u][id_anomalies] / s[id_anomalies][:, None]
                    P_v__u_y[u] *= P_v__u / P_v__u_y[u].mean(axis=0)
                    
                    n_corrections += 1
                    s = np.sum(P_v__u_y[u], axis=1)
                
                if self.verbose > 0:
                    print('\tCorrections done in '+str(n_corrections)+' iterations.')
        
        return(P_v__u_y, list_v__u)
        
    def _make_case(self,
                   initial_luc_layer,
                   final_luc_layer,
                   start_luc_layer):
        if self.verbose > 0:
            print('Make case...')
        
        self._J_X_u, self._X_u, self._v_u = self.case.make(initial_luc_layer, final_luc_layer)
        self._J_Y_u, self._Y_u = self.case.make(start_luc_layer)
        
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
    
    def _estimate_P_y__u_v(self, X_u=None, v_u=None, Y_u=None):
        
        if X_u is None:
            X_u = self._X_u
        if v_u is None:
            v_u = self._v_u
        if Y_u is None:
            Y_u = self._Y_u
        
        if self.verbose > 0:
            print('estimating P(y|u,v)')
        
        self._P_y__u_v = {}
        self._calibrated_transitions_u = {}
        
        for u in X_u.keys():
            
            self._calibrated_transitions_u[u] = []
            
            for idv, v in enumerate(self.case.params[u]['v']):
                if v != u:
                    if self.verbose > 0:
                        print('\tu='+str(u)+', v='+str(v))
                    
                    X_u_v = X_u[u][v_u[u] == v] 
                    
                    if X_u_v.shape[0] >= self.n_min and \
                        X_u_v.shape[0] / v_u[u].shape[0] > self.p_min:
                            
                            gkde = GKDE(
                                    h = 'scott',
                                    low_bounded_features = self._low_bounded_features_u[u],
                                    high_bounded_features = self._high_bounded_features_u[u],
                                    n_predict_max = self.n_predict_max,
                                    n_jobs = self.n_jobs,
                                    verbose = self.verbose - 1)
                            
                            if self.verbose > 0:
                                print('\tX_u_v shape : '+str(X_u_v.shape))
                                print('\tKDE fitting according to X_u_v')
                            
                            gkde.fit(X_u_v)
                            
                            if self.verbose > 0:
                                print('\th='+str(gkde._h))
                                print('\tP(y|u,v) estimation')
                                print('\tY shape : '+str(Y_u[u].shape))
                            
                            pdf = gkde.predict(Y_u[u])
                            if u in self._P_y__u_v.keys():
                                self._P_y__u_v[u] = np.hstack((self._P_y__u_v[u], pdf[:,None]))
                            else:
                                self._P_y__u_v[u] = pdf[:,None]
                                
                            self._calibrated_transitions_u[u].append(v)
                            
                    else:
                        if self.verbose > 0:
                            print('\tnot enough samples according to n_min and p_min.')
            
                if self.verbose > 0:
                    print('\t----------------')
            
        if self.verbose > 0:
            print('estimating P(y|u,v) done\n============================')
                    
    def _estimate_P_y__u(self, Y_u):
        if self.verbose > 0:
            print('estimating P(y|u)')
        
        self._P_y__u = {}
        
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
            
            gkde = GKDE(h = 'scott',
                        low_bounded_features = self._low_bounded_features_u[u],
                        high_bounded_features = self._high_bounded_features_u[u],
                        forbid_null_value = True,
                        n_predict_max = self.n_predict_max,
                        n_jobs = self.n_jobs,
                        verbose = self.verbose - 1)
            
            gkde.fit(Y_train)
            
            if self.verbose > 0:
                print('\th='+str(gkde._h))
                print('\tP(y|u) estimation')
            
            self._P_y__u[u] = gkde.predict(Y)[:, None]
        
            if self.verbose > 0:
                print('\t----------------')
        
        if self.verbose > 0:
            print('estimating P(y|u) done\n============================')
    
    # def _estimate_P_y__u_v(X_u, v_u, Y_u):
        