from ._tpe import TransitionProbabilityEstimator

# from ..density_estimation import GKDE

import numpy as np
# from ..density_estimation.density_estimatornsity_estimator import DensityEstimator, NullEstimator
from ..density_estimation import _methods as density_estimation_methods
from ..density_estimation import NullEstimator
from ..tools._console import title_heading

from .._base import Palette

from copy import deepcopy

class Bayes(TransitionProbabilityEstimator):
    """
    Bayes transition probability estimator.

    Parameters
    ----------
    density_estimator : {'gkde'} or DensityEstimator, default='gkde'
        Density estimator used for :math:`P(Y|u)`. If string, a new object is invoked with default parameters.

    n_corrections_max : int, default=1000
        Maximum number of corrections during the bayesian adjustment algorithm.

    log_computations : bool, default=False
        If ``True``, bayesian computations are made through log sums.

    verbose : int, default=0
        Verbosity level.

    """

    def __init__(self,
                 density_estimator='ekde',
                 n_corrections_max=1000,
                 n_fit_max=10**5,
                 log_computations=False,
                 # P_Y__v_layer=None,
                 verbose=0,
                 **kwargs):

        super().__init__(verbose=verbose)
        
        self.n_corrections_max = n_corrections_max
        self.n_fit_max = n_fit_max
        self.log_computations = log_computations
        self.density_estimator = density_estimator
        
        if type(density_estimator) is str:
            self.de = density_estimation_methods[density_estimator](verbose=verbose-1)
        else:
            self.de = density_estimator
        # self.P_Y__v_layer = P_Y__v_layer
    
    def __repr__(self):
        return ('Bayes')
    
    # def check(self, objects=None):
    #     """
    #     Check the unicity of objects.
    #     Notably, estimators uniqueness are checked to avoid malfunctioning during transition probabilities estimation.
    #     """
    #     if objects is None:
    #         objects = []
            
    #     if self.de in objects:
    #         raise(ValueError("DensityEstimator objects must be different."))
    #     else:
    #         objects.append(self.de)
        
    #     for cde in self.cde.values():
    #         if cde in objects:
    #             raise(ValueError("DensityEstimator objects must be different."))
    #         else:
    #             objects.append(cde)
    
    def fit(self,
            Z,
            W,
            bounds=None,
            **kwargs):
        """
        Fit the transition probability estimators. Only :math:`P(Y|u,v)` is
        concerned.

        Parameters
        ----------
        Z : array-like of shape (n_samples, n_features)
            Observed features data.
        W : array-like of shape (n_samples, n_final_states) of bool type
            One Hot Encoder of final states only
        bounds : list of string
            List of bounds info among {'left', 'right', 'both', 'none'}

        Returns
        -------
        self : TransitionProbabilityEstimator
            The fitted transition probability estimator object.

        """
        if self.verbose > 0:
            print('TPE fitting')
            print('Conditional Density Estimators fitting :')
        
        # set de params
        self.de.bounds = bounds
        
        # initialize and fit cde
        # if self.P_Y__v_layer is None:
        self.cde = []
        
        for i, w in enumerate(W.T):
            if w.sum() > 0:
                cde  = deepcopy(self.de)
                cde.set_params(bounds = bounds)
                
                Z_fit = Z[w]
                
                if Z_fit.shape[0] > self.n_fit_max:
                    Z_fit = Z_fit[np.random.choice(a=Z_fit.shape[0], 
                                                   size=self.n_fit_max,
                                                   replace=False)]
                
                if self.verbose > 0:
                    print('CDE #'+str(i)+' : '+str(cde))
                
                cde.fit(X=Z_fit)
            else:
                cde = NullEstimator()
                
                if self.verbose > 0:
                    print('CDE #'+str(i)+' : '+str(cde))
                    
            self.cde.append(cde)
        
        return (self)
    
    # def get_final_states(self):
    #     if self.P_Y__v_layer is None:
    #         return list(self.cde.keys())
    #     else:
    #         return self.P_Y__v_layer.final_states
    
    def compute(self,
                Y,
                P_v,
                P_Y = None,
                P_Y__v = None,
                return_P_Y = False,
                return_P_Y__v = False,
                **kwargs):
        """
        Estimates transition probability. Non estimated final states transition probabilities are filled to the null value.

        Parameters
        ----------
        transition_matrix : TransitionMatrix
            Land transition matrix with only one state in ``tm.palette_u``.

        Y : array-like of shape (n_samples, n_features)
            Samples to estimate transition probabilities

        Returns
        -------
        P_v__Y : ndarray of shape (n_samples, n_transitions)
            Transition probabilities for each studied final land use states ordered
            as ``transition_matrix.palette_v``.
        """
        
        if self.verbose > 0:
            print('TPE computing')

        # P(Y) estimation
        if P_Y is None:
            if self.verbose > 0:
                print('P(Z|u) estimation')
            P_Y = self.compute_P_Y(Y=Y)

        # P(Y|v) estimation
        if P_Y__v is None:
            if self.verbose > 0:
                print('P(Z|u,v) estimation')
            P_Y__v = self.compute_P_Y__v(Y=Y)
        
        # return P_Y__v
        
        # BAYES ADJUSTMENT PROCESS
        if self.verbose > 0:
            print('Bayes adjustment')
        P_v__Y = self.bayes_adjustment(P_Y__v=P_Y__v,
                                       P_Y=P_Y,
                                       P_v=P_v)
        
        # final_states = self.get_final_states()
                
        ret = [P_v__Y]
        
        if return_P_Y:
            ret.append(P_Y)
        if return_P_Y__v:
            ret.append(P_Y__v)
        
        if len(ret) == 1:
            return(ret[0])
        else:
            return ret

    def bayes_adjustment(self,
                         P_Y__v,
                         P_Y,
                         P_v):
                
        if self.log_computations == False:
            # if no log computation
            idx_not_null = P_Y[:,0] > 0
            P_v__Y = np.zeros_like(P_Y__v)
            P_v__Y[idx_not_null] = P_Y__v[idx_not_null] / P_Y[idx_not_null]

            P_v__Y_mean = P_v__Y.mean(axis=0)
            multiply_cols = P_v__Y_mean != 0
            P_v__Y[:, multiply_cols] *= P_v[multiply_cols] / P_v__Y_mean[multiply_cols]

            # P_v__Y *= P_v / P_v__Y.mean(axis=0)

            s = P_v__Y.sum(axis=1)

        else:
            # with log computation
            log_P_Y__v = np.zeros_like(P_Y__v)
            log_P_Y__v.fill(-np.inf)

            log_P_Y__v = np.log(P_Y__v, where=P_Y__v > 0, out=log_P_Y__v)

            log_P_Y = np.log(P_Y)

            log_P_v__Y = log_P_Y__v - log_P_Y
            log_P_v__Y -= np.log(np.mean(np.exp(log_P_v__Y), axis=0))
            log_P_v__Y += np.log(P_v)

            s = np.sum(np.exp(log_P_v__Y), axis=1)

        if np.sum(s > 1) > 0:
            if self.verbose > 0:
                s_sum = np.sum(s>1)
                print('Warning, '+str(s_sum)+'/'+str(s.size)+' ('+str(np.round(s_sum/s.size*100,2))+' %) uncorrect probabilities have been detected.')
                print('Some global probabilities may be to high.')
                print('For now, some corrections are made.')

            n_corrections = 0

            while np.sum(s > 1) > 0 and n_corrections < self.n_corrections_max:
                id_anomalies = s > 1

                if self.log_computations == False:
                    # if no log computation
                    P_v__Y[id_anomalies] = P_v__Y[id_anomalies] / \
                                           s[id_anomalies][:, None]

                    P_v__Y_mean = P_v__Y.mean(axis=0)
                    multiply_cols = P_v__Y_mean != 0
                    P_v__Y[:, multiply_cols] *= P_v[multiply_cols] / P_v__Y_mean[multiply_cols]

                    # P_v__Y *= P_v / P_v__Y.mean(axis=0)

                    s = np.sum(P_v__Y, axis=1)
                else:
                    # with log computation
                    log_P_v__Y[id_anomalies] = log_P_v__Y[id_anomalies] - np.log(s[id_anomalies][:, None])

                    log_P_v__Y -= np.log(np.mean(np.exp(log_P_v__Y), axis=0))
                    log_P_v__Y += np.log(P_v)

                    s = np.sum(np.exp(log_P_v__Y), axis=1)

                n_corrections += 1

            if self.verbose > 0:
                print('Corrections done in ' + str(n_corrections) + ' iterations.')

            if n_corrections == self.n_corrections_max:
                print(
                    'Warning : the P(v|Y) adjustment algorithm has reached the maximum number of loops. The n_corrections_max parameter should be increased.')

        if self.log_computations:
            P_v__Y = np.exp(log_P_v__Y)

        # last control to ensure s <= 1
        id_anomalies = s > 1
        P_v__Y[id_anomalies] = P_v__Y[id_anomalies] / \
                               s[id_anomalies][:, None]

        # avoid nan values
        P_v__Y = np.nan_to_num(P_v__Y)


        return(P_v__Y)

    def compute_P_Y(self, Y):
        # forbid_null_value is forced to True by default for this density estimator
        # self.de.set_params(forbid_null_value=True)

        if self.verbose > 0:
            print('Density estimator fitting')
        
        Y_fit = Y
        
        if Y_fit.shape[0] > self.n_fit_max:
            Y_fit = Y_fit[np.random.choice(a=Y_fit.shape[0], 
                                           size=self.n_fit_max,
                                           replace=False)]
        
        self.de.fit(Y_fit)

        # P(Y) estimation
        if self.verbose > 0:
            print('Density estimator predict')
        P_Y = self.de.predict(Y)[:, None]

        return (P_Y)

    def compute_P_Y__v(self, Y):

        # first, create a list of estimators according to palette_v order
        # if no estimator is informed, the NullEstimator is invoked.
        if self.verbose > 0:
            print('Conditionnal density estimators predict...')
        
        # estimate P(Y|u,v). Columns with no estimators are null columns.
        # if self.P_Y__v_layer is None:
        P_Y__v = np.vstack([cde.predict(Y) for cde in self.cde]).T
        # else:
            # P_Y__v = self.P_Y__v_layer.get_flat_proba(J=J)


        return (P_Y__v)
