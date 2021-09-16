#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy import ndimage
import numpy as np

from ..density_estimation import _methods
from ..density_estimation._density_estimator import DensityEstimator
from ._layer import _Layer
from . import State
from ..estimation import TransitionProbabilityEstimator
from ..tools import path_split
from ..definition import FeatureLayer

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
    verbose : int, default=None
        Verbosity level.
    """
    def __init__(self, 
                 features,
                 de = 'gkde',
                 isl_ratio = 1.0,
                 verbose = 0):
        
        self.features = features
        
        if isinstance(de, DensityEstimator):
            self.density_estimator = de
        elif de in _methods:
            self.density_estimator = _methods[de]()
        else:
            raise(ValueError('Unexpected de value.'))
                
        self.conditional_density_estimators = {}
        self.transition_patches = {}
        
        self.verbose = verbose
        
    def __repr__(self):
        return('land')
    
    def add_conditional_density_estimator(self, state, de='gkde'):
        """
        Add conditional density for a given final state.

        Parameters
        ----------
        state : State
            The final state.
        de : {'gkde'} or DensityEstimator, default='gkde'
            Density estimation for :math:`P(x|u,v)`.
                gkde : Gaussian Kernel Density Estimation method
        Returns
        -------
        self : Land
            The self object.
        """
        
        if isinstance(de, DensityEstimator):
            self.conditional_density_estimators[state] = de
        elif de in _methods:
            self.conditional_density_estimators[state] = _methods[de]()
        else:
            raise(ValueError('Unexpected de value.'))
        
        return(self)
        
    def add_transition_patches(self, 
                            state,
                            transition_patches):
        """
        Add transition patches for a given final state.

        Parameters
        ----------
        state : State
            The final state.
        transition_patches : TransitionPatches
            The transition patches

        Returns
        -------
        self : Land
            The self object.
        """
        
        self.transition_patches[state] = transition_patches
        
        return(self)
    
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
            
        luc_initial : LandUseCoverLayer
            The initial land use layer.
            
        luc_final : LandUseCoverLayer, default=None
            The final land use layer. Ignored if ``None``.
            
        region : LandUseCoverLayer, default=None
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
        data_luc_initial = luc_initial.get_data()
        
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
                if isinstance(info, _Layer):
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
            # just get data inside the region (because J is already inside)
            V = luc_final.get_data().flat[J]
        
    
        elements_to_return = [J]
    
        if explanatory_variables:
            elements_to_return.append(X)
        
        if luc_final is not None:
            elements_to_return.append(V)
        
        return(elements_to_return)

    def transition_probabilities(self,
                                 state,
                                 luc_initial,
                                 luc_final,
                                 luc_start,
                                 mask_calibration,
                                 mask_allocation,
                                 P_v,
                                 palette_v,
                                 distances_to_states_calibration={},
                                 distances_to_states_allocation={},
                                 path_prefix=None):
        """
        Computes transition probabilities

        Parameters
        ----------
        luc_initial : TYPE
            DESCRIPTION.
        luc_final : TYPE
            DESCRIPTION.
        region_calibration : TYPE
            DESCRIPTION.
        region_allocation : TYPE
            DESCRIPTION.
        distances_to_states : TYPE, optional
            DESCRIPTION. The default is {}.

        Returns
        -------
        None.

        """
        J_calibration, X, V = self.get_values(state = state,
                                luc_initial = luc_initial,
                                luc_final = luc_final,
                                mask = mask_calibration,
                                explanatory_variables=True,
                                distances_to_states=distances_to_states_calibration)
        
        estimator = TransitionProbabilityEstimator(density_estimator = self.density_estimator,
                                                   conditional_density_estimators = self.conditional_density_estimators)
        estimator.fit(X, V)
        
        J_allocation, Y = self.get_values(state = state,
                                luc_initial = luc_start,
                                mask = mask_allocation,
                                explanatory_variables=True,
                                distances_to_states=distances_to_states_allocation)
                
        P = estimator.transition_probability(Y, P_v, palette_v)
        
        if path_prefix is None:
            return(J_allocation, P)
        
        else:
            print(path_prefix,path_split(path_prefix, prefix=True))
            folder_path, file_prefix = path_split(path_prefix, prefix=True)
            
            
            for id_state, state in enumerate(palette_v):
                M = np.zeros(luc_start.get_data().shape)
                M.flat[J_allocation] = P[:, id_state]
                
                file_name = file_prefix + '_' + str(state.value) + '.tif'
                
                FeatureLayer(label=file_name,
                                data = M,
                                copy_geo = luc_start,
                                path = folder_path + '/' + file_name)
            
            return(True)

def _compute_distance(state, data, distances_to_states):
    v_matrix = (data == state.value).astype(int)
    distances_to_states[state] = ndimage.distance_transform_edt(1 - v_matrix)