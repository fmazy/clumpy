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

        super().__init__(n_corrections_max=1000,
                         log_computations=False,
                         verbose=verbose,
                         verbose_heading_level=verbose_heading_level)

        print(_methods)

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



    def add_conditional_density_estimator(self,
                                          state,
                                          density_estimator='gkde',
                                          P_v_min=5 * 10 ** (-5),
                                          n_samples_min=500):
        """
        Add conditional density for a given final state.

        Parameters
        ----------
        state : State
            The final state.

        density_estimator : {'gkde'} or DensityEstimator, default='gkde'
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

        if isinstance(density_estimator, DensityEstimator):
            self.conditional_density_estimators[state] = density_estimator
        elif density_estimator in _methods:
            self.conditional_density_estimators[state] = _methods[density_estimator](verbose=self.verbose - 1,
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
            state=None,
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
        self.density_estimator.set_params(low_bounded_features=low_bounded_features,
                                          high_bounded_features=high_bounded_features,
                                          low_bounds=low_bounds,
                                          high_bounds=high_bounds)

        for state_v, cde in self.conditional_density_estimators.items():
            if self.verbose > 0:
                print('state_v : ' + str(state_v))

            # set cde params
            cde.set_params(low_bounded_features=low_bounded_features,
                           high_bounded_features=high_bounded_features,
                           low_bounds=low_bounds,
                           high_bounds=high_bounds)

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

    def _compute_P_Y(self, Y, J=None):
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

        return (P_Y)

    def _compute_P_Y__v(self, Y, transition_matrix, J=None):
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

        return (P_Y__v)
