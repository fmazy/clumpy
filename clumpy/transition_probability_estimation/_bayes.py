from ._tpe import TransitionProbabilityEstimator

# from ..density_estimation import GKDE

import numpy as np
# from ..density_estimation._density_estimator import DensityEstimator, NullEstimator
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
                 density_estimator=None,
                 n_corrections_max=1000,
                 log_computations=False,
                 verbose=0,
                 verbose_heading_level=1,
                 **kwargs):

        super().__init__(n_corrections_max=1000,
                         log_computations=False,
                         verbose=verbose,
                         verbose_heading_level=verbose_heading_level)
        
        self.density_estimator = density_estimator
    
    def fit(self,
            X,
            V,
            v_initial=None,
            bounds=None):
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
        self._de = deepcopy(self.density_estimator)
        self._de.set_params(bounds = bounds)
        
        self._cde = {}
        
        list_v = np.unique(V)
        for v in list_v:
            if v != v_initial:
                self._cde[v] = deepcopy(self._de)
                self._cde[v].set_params(bounds = bounds)
                
                idx_v = V == v
                self._cde[v].fit(X=X[idx_v])
                
            else:
                self._cde[v] = NullEstimator()
        
        if self.verbose > 0:
            print('TPE fitting done.')

        return (self)

    def _compute_P_Y(self, Y):
        # forbid_null_value is forced to True by default for this density estimator
        # self._de.set_params(forbid_null_value=True)

        if self.verbose > 0:
            print('Density estimator fitting...')
        self._de.fit(Y)
        if self.verbose > 0:
            print('Density estimator fitting done.')

        # P(Y) estimation
        if self.verbose > 0:
            print('Density estimator predict...')
        P_Y = self._de.predict(Y)[:, None]
        if self.verbose > 0:
            print('Density estimator predict done.')

        return (P_Y)

    def _compute_P_Y__v(self, Y, list_v):

        # first, create a list of estimators according to palette_v order
        # if no estimator is informed, the NullEstimator is invoked.
        if self.verbose > 0:
            print('Conditionnal density estimators predict...')
            print('are concerned :')
        cde = []
        for v in list_v:
            if v in self._cde.keys():
                if self.verbose > 0:
                    print('v ' + str(v))
                cde.append(self._cde[v])
            else:
                cde.append(NullEstimator())

        # estimate P(Y|u,v). Columns with no estimators are null columns.
        P_Y__v = np.vstack([cde.predict(Y) for cde in cde]).T

        if self.verbose > 0:
            print('Conditional density estimators predict done.')

        return (P_Y__v)
