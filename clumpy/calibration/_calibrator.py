from ..estimation import RectKDE

import numpy as np

_estimators = {'rect_kde':RectKDE}

class Calibrator():
    def __init__(self,
                 estimator='rect_kde',
                 h = 'silverman',
                 p_min = 0.5*10**-4,
                 n_min = 500,
                 h_min = 0.1,
                 h_max = 1.0,
                 h_step = 0.01,
                 h_n_increasing = 10,
                 algorithm = 'kd_tree',
                 leaf_size = 30,
                 n_jobs = None,
                 verbose = 0):
        self.estimator = estimator
        self.h = h
        self.p_min = p_min
        self.n_min = n_min
        self.h_min = h_min
        self.h_max = h_max
        self.h_step = h_step
        self.h_n_increasing = h_n_increasing
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs=n_jobs
        self.verbose = verbose
    
    def fit_save(self,
                 X_u,
                 v_u=None,
                 bounded_features_u=[],
                 path_prefix=None):
        
        if type(path_prefix) is not str:
            raise(ValueError('Unexpected path prefix.'))
        
        if self.verbose > 0:
            print('Calibrator fitting')
            print('Estimator :',self.estimator)
            print('\n')
        for u in X_u.keys():    
            if v_u is not None:
                for e in np.unique(v_u[u]):
                    if e != u:
                        if self.verbose > 0:
                            print('=========')
                            print('Transition u='+str(u)+', v='+str(e))
                        X = X_u[u][v_u[u] == e]
                        
                        if X.shape[0]>0:
                            if X.shape[0] /  v_u[u].size > self.p_min and X.shape[0] >= self.n_min:
                                print('Estimation...')
                                kde = _estimators[self.estimator](h = self.h,
                                                                h_min = self.h_min,
                                                                h_max = self.h_max,
                                                                h_step = self.h_step,
                                                                h_n_increasing=self.h_n_increasing,
                                                                bounded_features=bounded_features_u[u],
                                                                algorithm=self.algorithm,
                                                                leaf_size=self.leaf_size,
                                                                n_jobs=self.n_jobs,
                                                                verbose=self.verbose-1)
                                kde.fit(X)
                                
                                kde.save(path_prefix+'kde_u'+str(u)+'_v'+str(e)+'.zip')
                            else:
                                if self.verbose>0:
                                    print('not enough observations for this transition')
                        else:
                            if self.verbose>0:
                                print('no observation for this transition')
            else:
                if X_u[u].shape[0] >= self.n_min:
                    print('Estimation')
                    
                    if self.verbose > 0:
                        print('=========')
                        print('state u='+str(u))
                    
                    kde = _estimators[self.estimator](h = self.h,
                                                    h_min = self.h_min,
                                                    h_max = self.h_max,
                                                    h_step = self.h_step,
                                                    h_n_increasing=self.h_n_increasing,
                                                    bounded_features=bounded_features_u[u],
                                                    algorithm=self.algorithm,
                                                    leaf_size=self.leaf_size,
                                                    n_jobs=self.n_jobs,
                                                    verbose=self.verbose-1)
                    kde.fit(X_u[u])
    
                    kde.save(path_prefix+'kde_u'+str(u)+'.zip')
                else:
                    if self.verbose>0:
                        print('not enough observations for this transition')

