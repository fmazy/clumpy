#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy import ndimage
import numpy as np

# base import
from ._layer import Layer, FeatureLayer, LandUseLayer
from ._state import State
from ._transition_matrix import TransitionMatrix

# Transition Probability Estimator
from ..transition_probability_estimation._tpe import TransitionProbabilityEstimator

# Features selection
from ..feature_selection._feature_selector import FeatureSelector

# Tools
from ..tools._path import path_split
from ..tools._console import title_heading

# Allocation
from ..allocation._allocator import Allocator
from ..allocation._compute_patches import compute_bootstrap_patches


class Land():
    """
    Land object which refers to a given initial state.

    Parameters
    ----------
    features : list(FeaturesLayer or State), default=[]
        List of features where a State means a distance layer to the corresponding state.

    transition_probability_estimator : TransitionProbabilityEstimator, default=None
        Transition probability estimator. If ``None``, fit, transition_probabilities and allocate are not available.

    feature_selection : FeatureSelection or list(FeatureSelection)
        List of features selection methods.

    fit_bootstrap_patches : bool, default=True
        If ``True``, make bootstrap patches when fitting.

    allocator : Allocator, default=None
        Allocator. If `None`, the allocation is not available.
    
    verbose : int, default=0
        Verbosity level.

    verbose_heading_level : int, default=1
        Verbose heading level for markdown titles. If ``0``, no markdown title are printed.
    """

    def __init__(self,
                 features=[],
                 transition_probability_estimator=None,
                 feature_selection=None,
                 fit_bootstrap_patches=False,
                 allocator=None,
                 verbose=0,
                 verbose_heading_level=1):

        # Transition probability estimator
        if transition_probability_estimator is not None and isinstance(transition_probability_estimator,
                                                                       TransitionProbabilityEstimator) == False:
            raise (TypeError(
                "Unexpected 'transition_probability_estimator. A 'TransitionProbabilityEstimator' object is expected."))
        self.transition_probability_estimator = transition_probability_estimator

        # features as a list
        if not isinstance(features, list):
            raise (TypeError("Unexpected 'features'. A list is expected."))
        self.features = features

        # Features selection
        self.feature_selection = feature_selection

        # fit bootstrap patches
        self.fit_bootstrap_patches = fit_bootstrap_patches

        # allocator
        self.allocator = allocator

        self.verbose = verbose
        self.verbose_heading_level = verbose_heading_level

    def __repr__(self):
        return 'land'

    def set_params(self,
                   **params):
        for key, param in params.items():
            setattr(self, key, param)

    def check(self):
        """
        Check the Land object.
        Notably, estimators uniqueness are checked to avoid malfunctioning during transition probabilities estimation.
        """

        self._check_density_estimators()
        self._check_feature_selectors()

    def _check_density_estimators(self, density_estimators=[]):
        """
        Check the density estimators uniqueness.
        """
        density_estimators = self.transition_probability_estimator._check(density_estimators=density_estimators)

        return (density_estimators)

    def _check_feature_selectors(self, feature_selectors=[]):
        """
        check the feature selectors uniqueness.
        """

        if isinstance(self.feature_selection, list):
            feature_selection = self.feature_selection
        else:
            feature_selection = [self.feature_selection]

        for fs in feature_selection:
            if fs in feature_selectors and fs is not None:
                raise (ValueError('The feature selection is already used. A new FeatureSelector must be invoked.'))
            feature_selectors.append(fs)

        return feature_selectors

    def get_values(self,
                   state,
                   lul_initial,
                   lul_final=None,
                   mask=None,
                   explanatory_variables=True,
                   distances_to_states={}):
        """
        Get values.

        Parameters
        ----------
        state : State
            The studied initial state.
            
        lul_initial : LandUseLayer or ndarray
            The initial land use layer.
            
        lul_final : LandUseLayer or ndarray, default=None
            The final land use layer. Ignored if ``None``.
            
        mask : MaskLayer, default=None
            The region mask layer. If ``None``, the whole area is studied.
            
        explanatory_variables : bool, default=True
            If ``True``, features values are returned.
            
        distances_to_states : dict(ndarray)
            A dict of ndarray distances_to_states to the State used as key.
            Usefull to avoid redondant distance computations.

        Returns
        -------
        J : ndarray of shape (n_samples,)
            The samples flat indexes.
        
        X : ndarray of shape (n_samples, n_features)
            Returned if ``explanatory_variables`` is ``True``. The features values.
        
        V : ndarray of shape (n_samples, n_features)
            Returned if ``lul_final`` is not ``None``. The final state values.

        """
        # initial data
        # the region is selected after the distance computation
        if isinstance(lul_initial, LandUseLayer):
            data_lul_initial = lul_initial.get_data().copy()
        else:
            data_lul_initial = lul_initial.copy()

        # selection according to the region.
        # one set -1 to non studied data
        # -1 is a forbiden state value.
        if mask is not None:
            data_lul_initial[mask.get_data() != 1] = -1

        # get pixels indexes whose initial states are u
        # J = ndarray_suitable_integer_type(np.where(initial_lul_layer.raster_.read(1).flat==u)[0])
        J = np.where(data_lul_initial.flat == state.value)[0]

        X = None
        if explanatory_variables:
            # create feature labels
            for info in self.features:
                # switch according z_type
                if isinstance(info, Layer):
                    # just get data
                    x = info.get_data().flat[J]

                elif isinstance(info, State):
                    # get distance data
                    # in this case, feature is a State object !
                    if info not in distances_to_states.keys():
                        _compute_distance(info, data_lul_initial, distances_to_states)
                    x = distances_to_states[info].flat[J]

                elif isinstance(info, int):
                    # get the corresponding state
                    state = lul_initial.palette.get(info)
                    # get distance data
                    # in this case, feature is a State object !
                    if state not in distances_to_states.keys():
                        _compute_distance(state, data_lul_initial, distances_to_states)
                    x = distances_to_states[state].flat[J]
                else:
                    raise (TypeError('Unexpected feature info : ' + str(info) + '.'))

                # if X is not yet defined
                if X is None:
                    X = x

                # else column stack
                else:
                    X = np.column_stack((X, x))

            # if only one feature, reshape X as a column
            if len(self.features) == 1:
                X = X[:, None]

        # if final lul layer
        if lul_final is not None:
            if isinstance(lul_final, LandUseLayer):
                data_lul_final = lul_final.get_data()
            else:
                data_lul_final = lul_final

            # just get data inside the region (because J is already inside)
            V = data_lul_final.flat[J]

        elements_to_return = [J]

        if explanatory_variables:
            elements_to_return.append(X)

        if lul_final is not None:
            elements_to_return.append(V)

        return elements_to_return

    def fit(self,
            state,
            lul_initial,
            lul_final,
            mask=None,
            distances_to_states={}):
        """
        Fit the land. Required for any further process.

        Parameters
        ----------
        state : State
            The initial state of this land.
            
        lul_initial : LandUseLayer
            The initial land use.
            
        lul_final : LandUseLayer
            The final land use.
            
        mask : MaskLayer, default = None
            The region mask layer. If ``None``, the whole area is studied.
        
        distances_to_states : dict(State:ndarray), default={}
            The distances matrix to key state. Used to improve performance.

        Returns
        -------
        self
        """

        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level) + 'Land ' + str(state) + ' fitting\n')

        if self.transition_probability_estimator is None:
            raise (ValueError('Transition probability estimator is expected for fitting.'))

        self._fit_tpe(state=state,
                      lul_initial=lul_initial,
                      lul_final=lul_final,
                      mask=mask,
                      distances_to_states=distances_to_states)

        if self.fit_bootstrap_patches:
            self.compute_bootstrap_patches(
                state=state,
                palette_v=self.transition_probability_estimator._palette_fitted_states,
                lul_initial=lul_initial,
                lul_final=lul_final,
                mask=mask)

        if self.verbose > 0:
            print('Land ' + str(state) + ' fitting done.\n')

        return self

    def _fit_tpe(self,
                 state,
                 lul_initial,
                 lul_final,
                 mask=None,
                 distances_to_states={}):
        """
        Fit the transition probability estimator
        """
        # GET VALUES
        J_calibration, X, V = self.get_values(state=state,
                                              lul_initial=lul_initial,
                                              lul_final=lul_final,
                                              mask=mask,
                                              explanatory_variables=True,
                                              distances_to_states=distances_to_states)

        # FEATURE SELECTORS
        # if only one object, make a list
        if isinstance(self.feature_selection, list):
            feature_selection = self.feature_selection
        elif self.feature_selection is None:
            feature_selection = []
        else:
            feature_selection = [self.feature_selection]

        # features_idx is used for boundaries parameters
        # it will be transformed as a X data
        # to get selected columns after all selectors.
        features_idx = np.arange(len(self.features))[None, :]

        for fs in feature_selection:
            if self.verbose > 0:
                print('Feature selecting : ' + str(fs) + '...')
            # check the type
            if not isinstance(fs, FeatureSelector):
                raise (TypeError(
                    "Unexpected 'feature_selection' type. Should be 'FeatureSelector' or 'list(FeatureSelector)'"))
            # fit and transform X
            X = fs.fit_transform(X=X)

            features_idx = fs.transform(features_idx)

            if self.verbose > 0:
                print('Feature selecting : ' + str(fs) + ' done.')

        features_idx = features_idx[0, :]

        # BOUNDARIES PARAMETERS
        low_bounded_features = []
        high_bounded_features = []
        low_bounds = []
        high_bounds = []
        for id_col, idx in enumerate(features_idx):
            if isinstance(self.features[idx], FeatureLayer):
                if self.features[idx].low_bound is not None:
                    # low_bounded_features takes as parameter the column id of
                    # bounded features AFTER feature selection !
                    # So, the right col index is id_col.
                    # idx is used to get the corresponding feature layer.
                    low_bounded_features.append(id_col)
                    low_bounds.append(self.features[idx].low_bound)

                if self.features[idx].high_bound is not None:
                    high_bounded_features.append(id_col)
                    high_bounds.append(self.features[idx].high_bound)
            if isinstance(self.features[idx], State) or isinstance(self.features[idx], int):
                low_bounded_features.append(id_col)
                low_bounds.append(0.0)

        # TRANSITION PROBABILITY ESTIMATOR
        self.transition_probability_estimator.fit(X,
                                                  V,
                                                  low_bounded_features=low_bounded_features,
                                                  high_bounded_features=high_bounded_features,
                                                  low_bounds=low_bounds,
                                                  high_bounds=high_bounds)

        return self

    def transition_matrix(self,
                          state,
                          lul_initial,
                          lul_final,
                          mask=None):
        """
        Compute the transition matrix.

        Parameters
        ----------
        state : State
            The initial state of this land.

        lul_initial : LandUseLayer
            The initial land use.

        lul_final : LandUseLayer
            The final land use.

        mask : MaskLayer, default = None
            The region mask layer. If ``None``, the whole area is studied.

        Returns
        -------
        tm : TransitionMatrix
            The computed transition matrix.
        """
        J, V = self.get_values(state=state,
                               lul_initial=lul_initial,
                               lul_final=lul_final,
                               mask=mask,
                               explanatory_variables=False)

        v_unique, n_counts = np.unique(V, return_counts=True)
        P_v = n_counts / n_counts.sum()
        P_v = P_v[None, :]

        palette_u = lul_initial.palette.extract(infos=[state])
        palette_v = lul_final.palette.extract(infos=v_unique)

        return (TransitionMatrix(M=P_v,
                                 palette_u=palette_u,
                                 palette_v=palette_v))

    def transition_probabilities(self,
                                 transition_matrix,
                                 lul,
                                 mask=None,
                                 distances_to_states={},
                                 path_prefix=None):
        """
        Computes transition probabilities.

        Parameters
        ----------
        transition_matrix : TransitionMatrix
            Land transition matrix with only one state in ``tm.palette_u``.

        lul : LandUseLayer
            The studied land use layer.

        mask : MaskLayer, default = None
            The region mask layer. If ``None``, the whole area is studied.

        distances_to_states : dict(State:ndarray), default={}
            The distances matrix to key state. Used to improve performance.

        path_prefix : str, default=None
            The path prefix to save result as ``path_prefix+'_'+str(state_v.value)+'.tif'.
            If None, the result is returned.
            Note that if ``path_prefix is not None``, ``lul`` must be LandUseLayer

        Returns
        -------
        J_allocation : ndarray of shape (n_samples,)
            Only returned if ``path_prefix=False``. Element indexes in the flattened
            matrix.

        P_v__u_Y : ndarray of shape (n_samples, len(palette_v))
            The transition probabilities of each elements. Columns are
            ordered as ``palette_v``.

        """

        # check if it is really a land transition matrix
        transition_matrix._check_land_transition_matrix()

        state = transition_matrix.palette_u.states[0]
        P_v, palette_v = transition_matrix.get_P_v(state)

        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level) + 'Land ' + str(state) + ' TPE\n')

        J, P_v__u_Y = self._compute_tpe(transition_matrix=transition_matrix,
                                        lul=lul,
                                        mask=mask,
                                        distances_to_states=distances_to_states)

        if self.verbose > 0:
            print('Land ' + str(state) + ' TPE done.\n')

        if path_prefix is None:
            return J, P_v__u_Y

        else:
            folder_path, file_prefix = path_split(path_prefix, prefix=True)

            for id_state, state_v in enumerate(palette_v):
                M = np.zeros(lul.get_data().shape)
                M.flat[J] = P_v__u_Y[:, id_state]

                file_name = file_prefix + '_' + str(state_v.value) + '.tif'

                FeatureLayer(label=file_name,
                             data=M,
                             copy_geo=lul,
                             path=folder_path + '/' + file_name)

            return True

    def _compute_tpe(self,
                     transition_matrix,
                     lul,
                     mask=None,
                     distances_to_states={},
                     save_P_Y__v=False,
                     save_P_Y=False,
                     return_Y=False):
        """
        Compute the transition probability estimation according to the given P_v
        """
        # check if it is really a land transition matrix
        transition_matrix._check_land_transition_matrix()

        state = transition_matrix.palette_u.states[0]

        # GET VALUES
        J_allocation, Y = self.get_values(state=state,
                                          lul_initial=lul,
                                          mask=mask,
                                          explanatory_variables=True,
                                          distances_to_states=distances_to_states)

        # FEATURES SELECTOR
        # if only one object, make a list
        if isinstance(self.feature_selection, list):
            feature_selection = self.feature_selection
        elif self.feature_selection is None:
            feature_selection = []
        else:
            feature_selection = [self.feature_selection]

        for fs in feature_selection:
            # transform Y according to the fitting.
            Y = fs.transform(X=Y)

        # TRANSITION PROBABILITY ESTIMATION
        P_v__u_Y = self.transition_probability_estimator.transition_probability(
            transition_matrix=transition_matrix,
            Y=Y,
            compute_P_Y__v=True,
            compute_P_Y=True,
            save_P_Y__v=save_P_Y__v,
            save_P_Y=save_P_Y)

        if return_Y:
            return J_allocation, P_v__u_Y, Y
        else:
            return J_allocation, P_v__u_Y

    def compute_bootstrap_patches(self,
                                  state,
                                  palette_v,
                                  lul_initial,
                                  lul_final,
                                  mask):
        """
        Compute Bootstrap patches

        """
        patches = compute_bootstrap_patches(state=state,
                                            palette_v=palette_v,
                                            land=self,
                                            lul_initial=lul_initial,
                                            lul_final=lul_final,
                                            mask=mask)

        self.allocator.set_params(patches=patches)

    def allocate(self,
                 transition_matrix,
                 lul,
                 lul_origin=None,
                 mask=None,
                 distances_to_states={},
                 path=None):
        """
        allocation.

        Parameters
        ----------
        transition_matrix : TransitionMatrix
            Land transition matrix with only one state in ``tm.palette_u``.

        lul : LandUseLayer or ndarray
            The studied land use layer. If ndarray, the matrix is directly edited (inplace).

        lul_origin : LandUseLayer
            Original land use layer. Usefull in case of regional allocations. If ``None``, the  ``lul`` layer is copied.

        mask : MaskLayer, default = None
            The region mask layer. If ``None``, the whole map is studied.

        distances_to_states : dict(State:ndarray), default={}
            The distances matrix to key state. Used to improve performance.

        path : str, default=None
            The path to save result as a tif file.
            If None, the allocation is only saved within `lul`, if `lul` is a ndarray.
            Note that if ``path`` is not ``None``, ``lul`` must be LandUseLayer.

        Returns
        -------
        lul_allocated : LandUseLayer
            Only returned if ``path`` is not ``None``. The allocated map as a land use layer.
        """
        # check if it is really a land transition matrix
        transition_matrix._check_land_transition_matrix()

        state = transition_matrix.palette_u.states[0]

        print(type(self.allocator))
        if not isinstance(self.allocator, Allocator):
            raise (ValueError("Unexpected 'allocator'. A clumpy.allocation.Allocator object is expected."))

        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level) + 'Land ' + str(state) + ' allocation\n')

        self.allocator.allocate(transition_matrix=transition_matrix,
                                land=self,
                                lul=lul,
                                lul_origin=lul_origin,
                                mask=mask,
                                distances_to_states=distances_to_states,
                                path=path)

        if self.verbose > 0:
            print('Land ' + str(state) + ' allocation done.\n')


def _compute_distance(state, data, distances_to_states):
    v_matrix = (data == state.value).astype(int)
    distances_to_states[state] = ndimage.distance_transform_edt(1 - v_matrix)
