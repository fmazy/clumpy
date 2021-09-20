#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy import ndimage
import numpy as np

# base import
from ._layer import Layer, FeatureLayer, LandUseLayer
from . import State

# Transition Probability Estimator
from ..transition_probability_estimation._tpe import TransitionProbabilityEstimator

# Features selection
from ..feature_selection._feature_selector import FeatureSelector

# Tools
from ..tools._path import path_split

# Allocation
from ..allocation._allocator import Allocator

class Land():
    """
    Land object which refers to a given initial state.

    Parameters
    ----------
    features : list(FeaturesLayer) or list(State)
        List of features where a State means a distance layer to the corresponding state.
    de : {'gkde'} or DensityEstimator, default='gkde'
        Density estimation for :math:`P(x|u)`.
            
            gkde : Gaussian Kernel Density Estimation method
            
    update_P_v__u_Y : bool, default=True
        If ``True``, :math:`P(v|u,Y)` is updated at each loop during the
        allocation process.
    
    n_allocation_try : int, default=10**3
        The maximum number of allocation loops.
    
    verbose : int, default=None
        Verbosity level.
    """
    def __init__(self, 
                 features,
                 transition_probability_estimator,
                 feature_selection = None,
                 allocator = 'unbiased',
                 verbose = 0):

        # Transition probability estimator
        if ~isinstance(transition_probability_estimator, TransitionProbabilityEstimator):
            raise(TypeError("Unexpected 'transition_probability_estimator. A 'TransitionProbabilityEstimator' object is expected."))
        self.transition_probability_estimator = transition_probability_estimator

        # features as a list
        if ~isinstance(features, list):
            raise(TypeError("Unexpected 'features'. A list is expected."))
        self.features = features

        # Features selection
        self.feature_selection = feature_selection

        # allocator
        self.allocator = allocator

        self.verbose = verbose
        
    def __repr__(self):
        return('land')

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
        density_estimators = self.transition_probability_estimator._check(density_estimators = density_estimators)

        return(density_estimators)

    def _check_feature_selectors(self, feature_selectors=[]):
        """
        check the feature selectors uniqueness.
        """

        if isinstance(self.feature_selection, list):
            feature_selection = self.feature_selection
        else:
            feature_selection = [self.feature_selection]

        for fs in feature_selection:
            if fs in feature_selectors:
                raise(ValueError('The feature selection is already used. A new FeatureSelector must be invoked.'))
            feature_selectors.append(fs)

        return(feature_selectors)


    def get_values(self,
                   state,
                   luc_initial,
                   luc_final = None,
                   mask = None,
                   explanatory_variables=True,
                   distances_to_states={}):
        """
        Get values.

        Parameters
        ----------
        state : State
            The studied initial state.
            
        luc_initial : LandUseLayer or ndarray
            The initial land use layer.
            
        luc_final : LandUseLayer or ndarray, default=None
            The final land use layer. Ignored if ``None``.
            
        mask : LandUseLayer, default=None
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
            Returned if ``luc_final`` is not ``None``. The final state values.

        """
        # initial data
        # the region is selected after the distance computation
        if isinstance(luc_initial, LandUseLayer):
            data_luc_initial = luc_initial.get_data().copy()
        else:
            data_luc_initial = luc_initial.copy()
        
        # selection according to the region.
        # one set -1 to non studied data
        # -1 is a forbiden state value.
        if mask is not None:
            data_luc_initial[mask.get_data() != 1] = -1
        
        # get pixels indexes whose initial states are u
        # J = ndarray_suitable_integer_type(np.where(initial_luc_layer.raster_.read(1).flat==u)[0])
        J = np.where(data_luc_initial.flat == state.value)[0]
        
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
                        _compute_distance(info, data_luc_initial, distances_to_states)
                    x = distances_to_states[info].flat[J]
                    
                else:
                    raise(TypeError('Unexpected feature.'))
                    
                # if X is not yet defined
                if X is None:
                    X = x
                
                # else column stack
                else:
                    X = np.column_stack((X, x))

            # if only one feature, reshape X as a column
            if len(self.features) == 1:
                X = X[:,None]
        
        # if final luc layer
        if luc_final is not None:
            if isinstance(luc_final, LandUseLayer):
                data_luc_final = luc_final.get_data()
            else:
                data_luc_final = luc_final
            
            # just get data inside the region (because J is already inside)
            V = data_luc_final.flat[J]
        
    
        elements_to_return = [J]
    
        if explanatory_variables:
            elements_to_return.append(X)
        
        if luc_final is not None:
            elements_to_return.append(V)
        
        return(elements_to_return)
    
    def fit(self,
            state,
            luc_initial,
            luc_final,
            mask=None,
            distances_to_states={}):
        """
        Fit the land. Required for any further process.

        Parameters
        ----------
        state : State
            The initial state of this land.
            
        luc_initial : LandUseLayer
            The initial land use.
            
        luc_final : LandUseLayer
            The final land use.
            
        mask : LandUseLayer, default = None
            The region mask layer. If ``None``, the whole area is studied.
        
        distances_to_states : dict(State:ndarray), default={}
            The distances matrix to key state. Used to improve performance.

        Returns
        -------
        self : Land
            The self object.
        """
        
        self._fit_tpe(state=state,
                     luc_initial=luc_initial,
                     luc_final=luc_final,
                     mask=mask,
                     distances_to_states=distances_to_states)
        
        return(self)
    
    def _fit_tpe(self,
                 state,
                 luc_initial,
                 luc_final,
                 mask=None,
                 distances_to_states={}):
        """
        Fit the transition probability estimator
        """
        # GET VALUES
        J_calibration, X, V = self.get_values(state = state,
                                luc_initial = luc_initial,
                                luc_final = luc_final,
                                mask = mask,
                                explanatory_variables=True,
                                distances_to_states=distances_to_states)

        # FEATURE SELECTORS
        # if only one object, make a list
        if isinstance(self.feature_selection, list):
            feature_selection = self.feature_selection
        else:
            feature_selection = [self.feature_selection]


        for fs in feature_selection:
            # check the type
            if ~isinstance(fs, FeatureSelector):
                raise(TypeError("Unexpected 'feature_selection' type. Should be 'FeatureSelector' or 'list(FeatureSelector)'"))
            # fit and transform X
            X = fs.fit_transform(X=X)

        # TRANSITION PROBABILITY ESTIMATOR
        self.transition_probability_estimator.fit(X, V)
        
        return(self)
    
    def _compute_tpe(self,
                     state,
                     luc,
                     P_v,
                     palette_v,
                     mask=None,
                     distances_to_states={}):
        """
        Compute the transition probability estimation according to the given P_v
        """
        # GET VALUES
        J_allocation, Y = self.get_values(state = state,
                                luc_initial = luc,
                                mask = mask,
                                explanatory_variables=True,
                                distances_to_states=distances_to_states)
                
        # FEATURES SELECTOR
        # if only one object, make a list
        if isinstance(self.feature_selection, list):
            feature_selection = self.feature_selection
        else:
            feature_selection = [self.feature_selection]

        for fs in feature_selection:
            # transform Y according to the fitting.
            Y = fs.transform(X=Y)

        # TRANSITION PROBABILITY ESTIMATION
        P_v__u_Y = self.transition_probability_estimator.transition_probability(Y, P_v, palette_v)
        
        return(J_allocation, P_v__u_Y)

    def transition_probabilities(self,
                                 state,
                                 luc,
                                 P_v,
                                 palette_v,
                                 mask=None,
                                 distances_to_states={},
                                 path_prefix=None):
        """
        Computes transition probabilities.

        Parameters
        ----------
        state : State
            The initial state of this land.

        luc : LandUseLayer
            The studied land use layer.

        P_v : ndarray of shape(len(palette_v,))
            The global transition probabilities :math:`P(v)`. The order corresponds to
            ``palette_v``

        palette_v : Palette
            The final state palette corresponding to ``P_v``.

        mask : LandUseLayer, default = None
            The region mask layer. If ``None``, the whole area is studied.

        distances_to_states : dict(State:ndarray), default={}
            The distances matrix to key state. Used to improve performance.

        path_prefix : str, default=None
            The path prefix to save result as ``path_prefix+'_' + str(state.value) + '.tif'.
            If None, the result is returned.
            Note that if ``path_prefix is not None``, ``luc`` must be LandUseLayer

        Returns
        -------
        J_allocation : ndarray of shape (n_samples,)
            Only returned if ``path_prefix=False``. Element indexes in the flattened
            matrix.

        P_v__u_Y : ndarray of shape (n_samples, len(palette_v))
            The transition probabilities of each elements. Columns are
            ordered as ``palette_v``.

        """

        J_allocation, P_v__u_Y = self._compute_tpe(state=state,
                                                   luc=luc,
                                                   P_v=P_v,
                                                   palette_v=palette_v,
                                                   mask=mask,
                                                   distances_to_states=distances_to_states)

        if path_prefix is None:
            return (J_allocation, P_v__u_Y)

        else:
            print(path_prefix, path_split(path_prefix, prefix=True))
            folder_path, file_prefix = path_split(path_prefix, prefix=True)

            for id_state, state in enumerate(palette_v):
                M = np.zeros(luc.get_data().shape)
                M.flat[J_allocation] = P_v__u_Y[:, id_state]

                file_name = file_prefix + '_' + str(state.value) + '.tif'

                FeatureLayer(label=file_name,
                             data=M,
                             copy_geo=luc,
                             path=folder_path + '/' + file_name)

            return (True)
    
    def allocate(self,
                   state,
                   P_v,
                   palette_v,
                   luc,
                   luc_origin=None,
                   mask=None,
                   distances_to_states={},
                   path=None):
        """
        allocation. luc can be both LandUseLayer and ndarray.

        Parameters
        ----------
        state : TYPE
            DESCRIPTION.
        P_v : TYPE
            DESCRIPTION.
        palette_v : TYPE
            DESCRIPTION.
        luc : TYPE
            DESCRIPTION.
        luc_origin : TYPE, optional
            DESCRIPTION. The default is None.
        mask : TYPE, optional
            DESCRIPTION. The default is None.
        distances_to_states : TYPE, optional
            DESCRIPTION. The default is {}.
        path : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if ~isinstance(self.allocator, Allocator):
            raise(ValueError("Unexpected 'allocator'. A clumpy.allocation.Allocator object is expected."))

        luc_data = self.allocator(state=state,
                           land=self,
                           P_v=P_v,
                           palette_v=palette_v,
                           luc=luc,
                           luc_origin=luc_origin,
                           mask=mask,
                           distances_to_states=distances_to_states,
                           path=path)



def _compute_distance(state, data, distances_to_states):
    v_matrix = (data == state.value).astype(int)
    distances_to_states[state] = ndimage.distance_transform_edt(1 - v_matrix)
