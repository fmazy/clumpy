from ._tpe import TransitionProbabilityEstimator
from .._base import Palette
from ..density_estimation import ASH
from ..density_estimation import bandwidth_selection
from ..density_estimation._whitening_transformer import _WhiteningTransformer
from tqdm import tqdm
import sparse
import numpy as np


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


class ASHBayes(TransitionProbabilityEstimator):
    def __init__(self,
                 h='scott',
                 q=10,
                 verbose=0,
                 verbose_heading_level=1):

        super().__init__(n_corrections_max=1000,
                         log_computations=False,
                         verbose=verbose,
                         verbose_heading_level=verbose_heading_level)

        self.h = h
        self.q = q
        self.palette_v = Palette()
        self.P_v_min = {}
        self.n_samples_min = {}

    def add_conditional_density_estimator(self,
                                          state,
                                          P_v_min=5 * 10 ** (-5),
                                          n_samples_min=500):
        self.palette_v.add(state)
        self.P_v_min[state] = P_v_min
        self.n_samples_min[state] = n_samples_min

    def fit(self,
            X,
            V,
            low_bounded_features=[],
            high_bounded_features=[],
            low_bounds=[],
            high_bounds=[]):

        self._n = X.shape[0]
        self._d = X.shape[1]

        self._wt = _WhiteningTransformer()
        X = self._wt.fit_transform(X)

        # BANDWIDTH SELECTION
        if type(self.h) is int or type(self.h) is float:
            self._h = float(self.h)

        elif type(self.h) is str:
            if self.h == 'scott' or self.h == 'silverman':
                self._h = 2.576 * bandwidth_selection.scotts_rule(X)
            else:
                raise (ValueError("Unexpected bandwidth selection method."))
        else:
            raise (TypeError("Unexpected bandwidth type."))

        if self.verbose > 0:
            print('Bandwidth selection done : h=' + str(self._h))

        # NORMALIZATION FACTOR
        self._normalization = 1 / (self._h ** self._d)

        # create a digitization for each shift
        self._digitizers = []
        # self._histograms = []

        self._histograms_v = {state_v: [] for state_v in self.palette_v}

        for i_shift in tqdm(range(self.q)):
            self._digitizers.append(Digitize(dx=self._h,
                                             shift=self._h / self.q * i_shift))

            X_digitized = self._digitizers[i_shift].fit_transform(X)

            uniques_u, inverse_u, counts_u = np.unique(X_digitized,
                                                 axis=0,
                                                 return_counts=True,
                                                 return_inverse=True)

            sparse_shape = [bins.size + 1 for bins in self._digitizers[i_shift]._bins]

            # self._histograms.append(sparse.COO(coords=uniques_u.T,
            #                                    data=counts_u / self._n,
            #                                    shape=sparse_shape))

            for state_v in self.palette_v:
                J_v = V == state_v.value
                inverse_u_v = inverse_u[J_v]
                uniques_inverse_u_v, counts_uniques_inverse_u_v = np.unique(inverse_u_v, return_counts=True)

                coords = uniques_u[uniques_inverse_u_v]
                data = counts_uniques_inverse_u_v / counts_uniques_inverse_u_v.sum()

                self._histograms_v[state_v].append(sparse.COO(coords = coords.T,
                                                              data = data,
                                                              shape = sparse_shape))

        self._palette_fitted_states = self.palette_v.copy()

    def _compute_all(self, Y, transition_matrix):
        state_u = transition_matrix.palette_u.states[0]

        Y = self._wt.transform(Y)

        m = Y.shape[0]

        P_Y = np.zeros(Y.shape[0])
        P_Y__v = np.zeros((Y.shape[0], len(transition_matrix.palette_v)))

        for i_shift in tqdm(range(self.q)):
            Y_digitized = self._digitizers[i_shift].transform(Y)

            uniques, inverse_indices, counts = np.unique(Y_digitized,
                                                         axis=0,
                                                         return_inverse=True,
                                                         return_counts=True)
            uniques = tuple(U for U in uniques.T)

            P_Y += counts[inverse_indices] / m

            for id_v, state_v in enumerate(transition_matrix.palette_v):
                if state_v in self.palette_v and state_v.value != state_u.value:
                    P_Y__v[:,id_v] += self._histograms_v[state_v][i_shift][uniques].todense()[inverse_indices]

        # Normalization
        P_Y *= self._normalization / self.q
        P_Y = P_Y[:,None]

        print('#null values ', np.sum(P_Y==0))

        P_Y__v *= self._normalization / self.q

        return(P_Y, P_Y__v)


    def _check(self, density_estimators=[]):
        return (density_estimators)
