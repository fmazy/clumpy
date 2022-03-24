from ._tpe import TransitionProbabilityEstimator
from .._base import FeatureLayer
from tqdm import tqdm
from .._base import Palette
import numpy as np
import pandas as pd

from ..density_estimation import _methods

class Digitize():
    def __init__(self, dx, shift=0):
        self.dx = dx
        self.shift = shift

    def fit(self, X):
        self._d = X.shape[1]
        self._bins = [np.arange(V.min() - self.dx + self.shift,
                                V.max() + self.dx + self.shift,
                                self.dx) for V in X.T]

        return (self)

    def transform(self, X):
        X = X.copy()
        for k in range(self._d):
            X[:, k] = np.digitize(X[:, k], bins=self._bins[k])
        return (X.astype(int))

    def fit_transform(self, X):
        self.fit(X)

        return (self.transform(X))

class Load(TransitionProbabilityEstimator):
    def __init__(self,
                 density_estimator='ash',
                 verbose=0,
                 verbose_heading_level=1):

        super().__init__(n_corrections_max=1000,
                         log_computations=False,
                         verbose=verbose,
                         verbose_heading_level=verbose_heading_level)

        self.tplayers = {}

        if isinstance(density_estimator, DensityEstimator):
            self.density_estimator = density_estimator
        elif density_estimator in _methods:
            self.density_estimator = _methods[density_estimator](verbose=self.verbose - 1,
                                                                 verbose_heading_level=self.verbose_heading_level + 1)
        else:
            raise (ValueError('Unexpected density_estimator value.'))

    def add_transition_probability_map(self,
                                       state,
                                       path):
        self.tplayers[state] = FeatureLayer(path=path)
        self._map_shape = self.tplayers[state].get_data().shape

    def fit(self,
            X,
            V,
            state=None,
            low_bounded_features=[],
            high_bounded_features=[],
            low_bounds=[],
            high_bounds=[]):
        return(1)

    def _compute_P_Y(self, Y, J):
        # ça vaudrait le coup d'avoir une fonction
        # fit_predict !
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

        self._map_P_Y = np.zeros(self._map_shape)
        self._map_P_Y.flat[J] = P_Y

        return (P_Y)

    def _compute_P_Y__v(self, Y, transition_matrix, J):
        state_u = transition_matrix.palette_u.states[0]

        P_Y__v = []
        for state_v in transition_matrix.palette_v:
            if state_v != state_u and state_v in self.tplayers.keys():
                P_Y__v.append(self.tplayers[state_v].get_data().flat[J])
            else:
                P_Y__v.append(np.zeros(Y.shape[0]))

        # estimate P(Y|u,v). Columns with no estimators are null columns.
        P_Y__v = np.vstack([p for p in P_Y__v]).T

        # according to bayes : P(X|u,v) # P(v|X,u)P(X|u)
        P_Y__v *= self._map_P_Y.flat[J][:,None]

        if self.verbose > 0:
            print('Conditional density estimators predict done.')

        return (P_Y__v)

    def _check(self, density_estimators=[]):
        """
        Check the density estimators uniqueness.
        """
        # density estimator check.
        if self.density_estimator in density_estimators:
            raise (ValueError('The density estimator is already used. A new DensityEstimator must be invoked.'))
        density_estimators.append(self.density_estimator)

        return (density_estimators)
