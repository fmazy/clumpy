from ._tpe import TransitionProbabilityEstimator

from ..density_estimation import GKDE

import numpy as np
from ..density_estimation._density_estimator import DensityEstimator, NullEstimator
from ..density_estimation import _methods
from ..tools._console import title_heading

from .._base import Palette

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
                 density_estimator='gkde',
                 n_corrections_max=1000,
                 log_computations=False,
                 verbose=0,
                 verbose_heading_level=1):

        super().__init__(verbose=verbose,
                         verbose_heading_level=verbose_heading_level)

        if isinstance(density_estimator, DensityEstimator):
            self.density_estimator = density_estimator
        elif density_estimator in _methods:
            self.density_estimator = _methods[density_estimator](verbose=self.verbose - 1,
                                                                 verbose_heading_level=self.verbose_heading_level + 1)
        else:
            raise (ValueError('Unexpected density_estimator value.'))

        self.conditional_density_estimators = {}

        self.P_v_min = {}
        self.n_samples_min = {}

        self.n_corrections_max = n_corrections_max
        self.log_computations = log_computations

    def add_conditional_density_estimator(self,
                                          state,
                                          density_estimation='gkde',
                                          P_v_min=5 * 10 ** (-5),
                                          n_samples_min=500):
        """
        Add conditional density for a given final state.

        Parameters
        ----------
        state : State
            The final state.

        density_estimation : {'gkde'} or DensityEstimator, default='gkde'
            Density estimation for :math:`P(x|u,v)`.
                gkde : Gaussian Kernel Density Estimation method

        P_v_min : float, default=5*10**(-5)
        Minimum global probability to learn.

        n_samples_min : int, default=500
            Minimum number of samples to learn.

        Returns
        -------
        self : Land
            The self object.
        """

        if isinstance(density_estimation, DensityEstimator):
            self.conditional_density_estimators[state] = density_estimation
        elif density_estimation in _methods:
            self.conditional_density_estimators[state] = _methods[density_estimation](verbose=self.verbose - 1,
                                                                      verbose_heading_level=self.verbose_heading_level + 1)
        else:
            raise (ValueError('Unexpected de value.'))

        self.P_v_min[state] = P_v_min
        self.n_samples_min[state] = n_samples_min

        return (self)

    def _check(self, density_estimators=[]):
        """
        Check the density estimators uniqueness.
        """
        # density estimator check.
        if self.density_estimator in density_estimators:
            raise (ValueError('The density estimator is already used. A new DensityEstimator must be invoked.'))
        density_estimators.append(self.density_estimator)

        # conditional density estimator check.
        for state, cde in self.conditional_density_estimators.items():
            if cde in density_estimators:
                raise (ValueError('The conditional density estimator for the state ' + str(
                    state) + ' is already used. A new DensityEstimator must be invoked.'))
            density_estimators.append(cde)

        return (density_estimators)

    def fit(self,
            X,
            V,
            low_bounded_features=[],
            high_bounded_features=[],
            low_bounds=[],
            high_bounds=[]):
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
        self.density_estimator.set_params(low_bounded_features = low_bounded_features,
                                          high_bounded_features = high_bounded_features,
                                          low_bounds = low_bounds,
                                          high_bounds = high_bounds)

        for state_v, cde in self.conditional_density_estimators.items():
            if self.verbose > 0:
                print('state_v : ' + str(state_v))

            # set cde params
            cde.set_params(low_bounded_features = low_bounded_features,
                           high_bounded_features = high_bounded_features,
                           low_bounds = low_bounds,
                           high_bounds = high_bounds)

            # select X_v
            idx_v = V == state_v.value

            # check fitting conditions
            if np.mean(idx_v) > self.P_v_min[state_v] and np.sum(idx_v) > self.n_samples_min[state_v]:
                # Density estimation fit
                cde.fit(X[idx_v])

                self._palette_fitted_states.add(state_v)
            else:
                # fitting conditions are not satisfying
                # change the density estimator to the null estimator.
                self.conditional_density_estimators[state_v] = NullEstimator()

        if self.verbose > 0:
            print('TPE fitting done.')

        return (self)

    def transition_probability(self,
                               transition_matrix,
                               Y,
                               id_J=None,
                               compute_P_Y__v=True,
                               compute_P_Y = True,
                               save_P_Y__v=False,
                               save_P_Y=False):
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
        if id_J is None:
            id_J = np.ones(Y.shape[0]).astype(bool)

        # check if it is really a land transition matrix
        transition_matrix._check_land_transition_matrix()

        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level) + 'TPE computing')

        # P(Y) estimation
        if compute_P_Y:
            P_Y = self._compute_P_Y(Y=Y[id_J])
        else:
            P_Y = self.P_Y[id_J]

        # P(Y|v) estimation
        if compute_P_Y__v:
            P_Y__v = self._compute_P_Y__v(Y=Y[id_J], transition_matrix=transition_matrix)
        else:
            P_Y__v = self.P_Y__v[id_J]

        # BAYES ADJUSTMENT PROCESS
        P_v__Y = self._bayes_adjustment(P_Y__v=P_Y__v,
                                        P_Y=P_Y,
                                        transition_matrix=transition_matrix)

        if self.verbose > 0:
            print('TPE computing done.')

        if save_P_Y__v:
            self.P_Y__v = P_Y__v

        if save_P_Y:
            self.P_Y = P_Y

        return (P_v__Y)

    def _compute_P_Y(self, Y):
        # forbid_null_value is forced to True by default for this density estimator
        self.density_estimator.set_params(forbid_null_value=True)

        if self.verbose > 0:
            print('Density estimator fitting...')
        self.density_estimator.fit(Y)
        if self.verbose > 0:
            print('Density estimator fitting done.')

        # P(Y) estimation
        if self.verbose > 0:
            print('Density estimator predict...')
        P_Y = self.density_estimator.predict(Y)[:, None]
        if self.verbose > 0:
            print('Density estimator predict done.')

        return(P_Y)

    def _compute_P_Y__v(self, Y, transition_matrix):
        state_u = transition_matrix.palette_u.states[0]

        # first, create a list of estimators according to palette_v order
        # if no estimator is informed, the NullEstimator is invoked.
        if self.verbose > 0:
            print('Conditionnal density estimators predict...')
            print('are concerned :')
        conditional_density_estimators = []
        for state_v in transition_matrix.palette_v:
            if state_v != state_u and state_v in self.conditional_density_estimators.keys():
                if self.verbose > 0:
                    print('state_v ' + str(state_v))
                conditional_density_estimators.append(self.conditional_density_estimators[state_v])
            else:
                conditional_density_estimators.append(NullEstimator())

        # estimate P(Y|u,v). Columns with no estimators are null columns.
        P_Y__v = np.vstack([cde.predict(Y) for cde in conditional_density_estimators]).T

        if self.verbose > 0:
            print('Conditional density estimators predict done.')

        return(P_Y__v)

    def _bayes_adjustment(self,
                          P_Y__v,
                          P_Y,
                          transition_matrix):

        P_v = transition_matrix.M[0,:]

        if self.log_computations == False:
            # if no log computation
            P_v__Y = P_Y__v / P_Y

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
        state_u = transition_matrix.palette_u.states[0]
        id_state_u = transition_matrix.palette_v.get_id(state_u)
        P_v__Y[:, id_state_u] = 1 - np.delete(P_v__Y, id_state_u, axis=1).sum(axis=1)

        return(P_v__Y)

#def marginals(self):


