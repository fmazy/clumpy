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
        
        if self.verbose > 0:
            print('Make case...')
        
        J_X_u, X_u, v_u = self.case.make(initial_luc_layer, final_luc_layer)
        J_Y_u, Y_u = self.case.make(start_luc_layer)
        
        if self.verbose > 0:
            print('done')
        
        self._get_low_high_bounded_features()
        
        self._estimate_P_y__u_v(X_u, v_u, Y_u)
        
        self._estimate_P_y__u(Y_u)
        
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
    
    def _estimate_P_y__u_v(self, X_u, v_u, Y_u):
        if self.verbose > 0:
            print('estimating P(y|u,v)')
        
        self._P_y__u_v = {}
        
        for u in X_u.keys():
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
        