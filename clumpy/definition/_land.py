#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy import ndimage
import numpy as np

from ..density_estimation import _methods
from ..density_estimation._density_estimator import DensityEstimator
from ._layer import _Layer
from . import State

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
            self.density_estimation = de
        elif de in _methods:
            self.density_estimation = _methods[de]()
        else:
            raise(ValueError('Unexpected de value.'))
                
        self.conditional_density_estimation = {}
        self.transition_patches = {}
        
        self.verbose = verbose
        
    def __repr__(self):
        return('land')
    
    def add_conditional_density_estimation(self, state, de='gkde'):
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
            self.conditional_density_estimation[state] = de
        elif de in _methods:
            self.conditional_density_estimation[state] = _methods[de]()
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
                   region = None,
                   explanatory_variables=True,
                   distances={}):
        """
        Get values.

        """
        # initial data
        # the region is selected after the distance computation
        data_luc_initial = luc_initial.get_data()
                    
        # get pixels indexes whose initial states are u
        # J = ndarray_suitable_integer_type(np.where(initial_luc_layer.raster_.read(1).flat==u)[0])
        J = np.where(data_luc_initial.flat == state.value)[0]
        
        # selection according to the region.
        if region is not None:
            J = J[region.get_data().flat == 1]
        
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
                    if info not in distances.keys():
                        _compute_distance(info, data_luc_initial, distances)
                    x = distances[info].flat[J]
                    
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
        
    
        if explanatory_variables:
            # if no final luc layer
            if luc_final is None:
                return(J, X)
            
            else:
                return(J, X, V)
        else:
            if luc_final is None:
                return(J)
            
            else:
                return(J, V)

def _compute_distance(state, data, distances):
    v_matrix = (data == state.value).astype(int)
    distances[state] = ndimage.distance_transform_edt(1 - v_matrix)