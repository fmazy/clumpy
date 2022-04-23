from ._tpe import TransitionProbabilityEstimator

# from ..density_estimation import GKDE

import numpy as np
# from ..density_estimation.density_estimatornsity_estimator import DensityEstimator, NullEstimator
from ..density_estimation import _methods, NullEstimator
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

    verbose_heading_level : int, default=1
        Verbose heading level for markdown titles. If ``0``, no markdown title are printed.

    """

    def __init__(self,
                 initial_state,
                 density_estimator=None,
                 n_corrections_max=1000,
                 log_computations=False,
                 verbose=0,
                 verbose_heading_level=1,
                 **kwargs):

        super().__init__(initial_state=initial_state,
                         verbose=verbose,
                         verbose_heading_level=verbose_heading_level)
        
        self.n_corrections_max = n_corrections_max
        self.log_computations = log_computations
        
        self.de = density_estimator
    
    def __repr__(self):
        return ('Bayes')
    
    def check(self, objects=None):
        """
        Check the unicity of objects.
        Notably, estimators uniqueness are checked to avoid malfunctioning during transition probabilities estimation.
        """
        if objects is None:
            objects = []
            
        if self.de in objects:
            raise(ValueError("DensityEstimator objects must be different."))
        else:
            objects.append(self.de)
        
        for cde in self.cde.values():
            if cde in objects:
                raise(ValueError("DensityEstimator objects must be different."))
            else:
                objects.append(cde)
    
    def fit(self,
            X,
            V,
            bounds=None,
            **kwargs):
        """
        Fit the transition probability estimators. Only :math:`P(Y|u,v)` is
        concerned.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Observed features data.
        V : array-like of shape (n_samples,) of int type
            The final land use state values. Only studied final v should appear.
        u : int
            The initial land use state.

        Returns
        -------
        self : TransitionProbabilityEstimator
            The fitted transition probability estimator object.

        """
        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level) + 'TPE fitting')
            print('Conditional density estimators fitting :')

        self._palette_fitted_states = Palette()
        
        # set de params
        self.de.set_params(bounds = bounds)
        
        self.cde = {}
        
        final_states = list(np.unique(V))
        
        if self.initial_state not in final_states:
            final_states.append(self.initial_states)
        
        for v in final_states:
            if v != self.initial_state:
                self.cde[v] = deepcopy(self.de)
                self.cde[v].set_params(bounds = bounds)
                
                idx_v = V == v
                self.cde[v].fit(X=X[idx_v])
                
            else:
                self.cde[v] = NullEstimator()
        
        if self.verbose > 0:
            print('TPE fitting done.')

        return (self)
    
    def get_final_states(self):
        return(list(self.cde.keys()))
    
    def transition_probabilities(self,
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
            print(title_heading(self.verbose_heading_level) + 'TPE computing')

        # P(Y) estimation
        if P_Y is None:
            P_Y = self.compute_P_Y(Y=Y)

        # P(Y|v) estimation
        if P_Y__v is None:
            P_Y__v = self.compute_P_Y__v(Y=Y)

        # BAYES ADJUSTMENT PROCESS
        P_v__Y = self.bayes_adjustment(P_Y__v=P_Y__v,
                                        P_Y=P_Y,
                                        P_v=P_v)
        
        final_states = self.get_final_states()
        
        if self.verbose > 0:
            print('TPE computing done.')
        
        ret = [P_v__Y, final_states]
        
        if return_P_Y:
            ret.append(P_Y)
        if return_P_Y__v:
            ret.append(P_Y__v)
            
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

        # compute the non transited column
        final_states = self.get_final_states()
        id_initial_state = final_states.index(self.initial_state)
        P_v__Y[:, id_initial_state] = 1 - np.delete(P_v__Y, id_initial_state, axis=1).sum(axis=1)

        return(P_v__Y)

    def compute_P_Y(self, Y):
        # forbid_null_value is forced to True by default for this density estimator
        # self.de.set_params(forbid_null_value=True)

        if self.verbose > 0:
            print('Density estimator fitting...')
        self.de.fit(Y)
        if self.verbose > 0:
            print('Density estimator fitting done.')

        # P(Y) estimation
        if self.verbose > 0:
            print('Density estimator predict...')
        P_Y = self.de.predict(Y)[:, None]
        if self.verbose > 0:
            print('Density estimator predict done.')

        return (P_Y)

    def compute_P_Y__v(self, Y):

        # first, create a list of estimators according to palette_v order
        # if no estimator is informed, the NullEstimator is invoked.
        if self.verbose > 0:
            print('Conditionnal density estimators predict...')
        
        # estimate P(Y|u,v). Columns with no estimators are null columns.
        P_Y__v = np.vstack([cde.predict(Y) for cde in self.cde.values()]).T

        if self.verbose > 0:
            print('Conditional density estimators predict done.')

        return (P_Y__v)
