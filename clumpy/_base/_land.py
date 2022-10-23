#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from time import time
from copy import deepcopy
from scipy import ndimage

# base import
# from ..layer import Layer, FeatureLayer, LandUseLayer, MaskLayer, ProbaLayer
from ..layer._proba_layer import create_proba_layer
from ._state import State
from ._transition_matrix import TransitionMatrix, load_transition_matrix

# Transition Probability Estimator
from ..density_estimation import _methods as _density_estimation_methods
from ..transition_probability_estimation._tpe import TransitionProbabilityEstimator
from ..transition_probability_estimation import Bayes

# Tools
from ..tools._path import path_split
from ..tools._console import title_heading
from ..tools._funcs import extract_parameters

# Allocation
from ..allocation._allocator import Allocator
from ..allocation import _methods as _allocation_methods

from ..transition_probability_estimation import _methods as transition_probability_estimation_methods
from ..ev_selection import _methods as ev_selection_methods
from ..ev_selection import EVSelectors
from ..allocation import _methods as allocation_methods
from ..patch import _methods as patch_methods

import logging
logger = logging.getLogger('clumpy')

class Land():
    """
    Land object which refers to a given initial state.

    Parameters
    ----------
    features : list(featuresLayer or State), default=[]
        List of features where a State means a distance layer to the corresponding state.

    transition_probability_estimator : TransitionProbabilityEstimator, default=None
        Transition probability estimator. If ``None``, fit, transition_probabilities and allocate are not available.

    feature_selector : featureSelection or list(featureSelection)
        List of features selection methods.

    fit_bootstrap_patches : bool, default=False
        If ``True``, make bootstrap patches when fitting.

    allocator : Allocator, default=None
        Allocator. If `None`, the allocation is not available.
    
    verbose : int, default=0
        Verbosity lfeatureel.

    verbose_heading_level : int, default=1
        Verbose heading lfeatureel for markdown titles. If ``0``, no markdown title are printed.
    """

    def __init__(self,
                 state,
                 final_states,
                 transition_probability_estimator=None,
                 ev_selectors=None,
                 allocator=None,
                 patcher=None,
                 verbose=0,
                 verbose_heading_level=1):
        
        # state
        self.state = state
        self.final_states = final_states
        
        if type(transition_probability_estimator) is str:
            self.transition_probability_estimator = transition_probability_estimation_methods[transition_probability_estimator]()
        else:
            self.transition_probability_estimator = transition_probability_estimator
        
        if type(ev_selectors) is str:
            self.ev_selectors = EVSelectors(selectors={v:ev_selection_methods[ev_selectors]() for v in final_states})
            
        else:
            self.ev_selectors = ev_selectors
        
        if type(allocator) is str:
            self.allocator = allocation_methods[allocator]()
        else:
            self.allocator = allocator 
        
        if type(patcher) is str:
            self.patcher = [patch_methods[patcher]() for v in final_states]
        else:
            self.patcher = patcher
        
        self.verbose = verbose
        self.verbose_heading_level = verbose_heading_level
        
    def __repr__(self):
        return 'Land()'
    
    
    
    # def set_params(self,
    #                **params):
    #     for key, param in params.items():
    #         setattr(self, key, param)
    
    # def check(self, objects=None):
    #     """
    #     Check the unicity of objects.
    #     Notably, estimators uniqueness are checked to avoid malfunctioning during transition probabilities estimation.
    #     """
    #     if objects is None:
    #         objects = []
            
    #     if self.calibrator in objects:
    #         raise(ValueError("Calibrator objects must be different."))
    #     else:
    #         objects.append(self.calibrator)
        
    #     self.calibrator.check(objects=objects)
       
    # def set_lul(self, lul, kind):
    #     self.lul[kind] = lul
    #     return(self)
    
    # def get_lul(self, kind):
    #     if kind not in self.lul.keys():
    #         return(self.region.get_lul(kind))
    #     else:
    #         return(self.lul[kind])
        
    # def set_mask(self, mask, kind):
    #     self.mask[kind] = mask
    #     return(self)
    
    # def get_mask(self, kind):
    #     if kind not in self.mask.keys():
    #         if self.region is None:
    #             return(None)
    #         else:
    #             return(self.region.get_mask(kind))
    #     else:
    #         return(self.mask[kind])
    
    # def set_transition_matrix(self, tm):
    #     self.transition_matrix = tm
    #     return(self)
    
    # def get_transition_matrix(self):
    #     if self.transition_matrix is None:
    #         return(self.region.get_transition_matrix())
    #     else:
    #         return(self.transition_matrix)
    
    # def set_features(self, features):
    #     self.features = features
    #     return(self)
    
    # def get_features(self):
    #     if self.features is None:
    #         return(self.region.get_features())
    #     else:
    #         return(self.features)

#     def fit(self):
#         """
#         Fit the land. Required for any further process.

#         Parameters
#         ----------
#         state : State
#             The initial state of this land.
            
#         lul_initial : LandUseLayer
#             The initial land use.
            
#         lul_final : LandUseLayer
#             The final land use.
            
#         mask : MaskLayer, default = None
#             The region mask layer. If ``None``, the whole area is studied.
        
#         distances_to_states : dict(State:ndarray), default={}
#             The distances matrix to key state. Used to improve performance.

#         Returns
#         -------
#         self
#         """
        
#         if self.verbose > 0:
#             print(title_heading(self.verbose_heading_level) + 'Land ' + str(self.state) + ' fitting\n')
        
#         self.calibrator.fit(lul_initial=self.get_lul('initial'),
#                lul_final=self.get_lul('final'),
#                features=self.get_features(),
#                mask=self.get_mask('calibration'))
                        

#         return self
    
#     def transition_probabilities(self, 
#                                  effective_transitions_only=True):        
        
#         if self.verbose > 0:
#             print(title_heading(self.verbose_heading_level) + 'Land ' + str(self.state) + ' TPE\n')
        
#         J, P_v__u_Y, final_states = self.calibrator.transition_probabilities(
#             lul=self.get_lul('start'),
#             tm=self.get_transition_matrix(),
#             features=self.get_features(),
#             mask = self.get_mask('allocation'),
#             effective_transitions_only=effective_transitions_only)
        
#         return J, P_v__u_Y, final_states
    
#     def transition_probabilities_layer(self, 
#                                        effective_transitions_only=True):

        
#         J, P, final_states = self.transition_probabilities(
#             effective_transitions_only=effective_transitions_only)
        
#         lul = self.get_lul('start')
#         proba_layer = create_proba_layer(J=J,
#                                          P=P,
#                                          final_states=final_states,
#                                          shape=lul.shape,
#                                          geo_metadata=lul.geo_metadata)
                
#         return(proba_layer)

#     # def allocate(self,
#     #              J,
#     #              P,
#     #              final_states,
#     #              lul='start',
#     #              lul_origin=None,
#     #              mask=None):
#     #     """
#     #     allocation.

#     #     Parameters
#     #     ----------
#     #     transition_matrix : TransitionMatrix
#     #         Land transition matrix with only one state in ``tm.palette_u``.

#     #     lul : LandUseLayer or ndarray
#     #         The studied land use layer. If ndarray, the matrix is directly edited (inplace).

#     #     lul_origin : LandUseLayer
#     #         Original land use layer. Usefull in case of regional allocations. If ``None``, the  ``lul`` layer is copied.

#     #     mask : MaskLayer, default = None
#     #         The region mask layer. If ``None``, the whole map is studied.

#     #     distances_to_states : dict(State:ndarray), default={}
#     #         The distances matrix to key state. Used to improve performance.

#     #     path : str, default=None
#     #         The path to save result as a tif file.
#     #         If None, the allocation is only saved within `lul`, if `lul` is a ndarray.
#     #         Note that if ``path`` is not ``None``, ``lul`` must be LandUseLayer.

#     #     path_prefix_transition_probabilities : str, default=None
#     #         The path prefix to save transition probabilities.

#     #     Returns
#     #     -------
#     #     lul_allocated : LandUseLayer
#     #         Only returned if ``path`` is not ``None``. The allocated map as a land use layer.
#     #     """
        
#     #     if isinstance(lul, str):
#     #         lul = self.get_lul(lul).copy()
                
#     #     self.allocator.allocate(J=J,
#     #                             P=P,
#     #                             final_states=final_states,
#     #                             lul=lul,
#     #                             mask=mask,
#     #                             lul_origin=lul_origin)
        
#     #     return(lul)
    
#     def allocate(self,
#                  lul='start',
#                  lul_origin=None):
        
#         if self.verbose > 0:
#             print(title_heading(self.verbose_heading_level) + 'Land ' + str(self.state) + ' Allocation\n')
        
#         if isinstance(lul, str):
#             lul = self.get_lul(lul).copy()
        
#         if lul_origin is None:
#             lul_origin = lul.copy()
        
#         lul, proba_layer = self.allocator.allocate(
#             lul=lul,
#             tm=self.get_transition_matrix(),
#             features=self.get_features(),
#             lul_origin=lul_origin,
#             mask=self.get_mask('allocation'))    
        
#         return lul, proba_layer
    
# # def make(self, palette, **params):
#     #     # features
#     #     features = []
#     #     if 'features' in params.keys():
#     #         for feature_params in params['features']:
#     #             if feature_params['type'] == 'layer':
#     #                 fp = extract_parameters(FeatureLayer, feature_params)
#     #                 features.append(FeatureLayer(**fp))
#     #             elif feature_params['type'] == 'distance':
#     #                 if feature_params['state'] != self.state.value:
#     #                     features.append(palette._get_by_value(feature_params['state']))
        
#     #     self.features = features
        
#     #     # feature selection
#     #     if 'feature_selection' in params.keys():
#     #         if isinstance(params['feature_selection'], int):
#     #             self.feature_selection = params['feature_selection']
#     #         else:
#     #             self.feature_selection = -1
        
#     #     # transition matrix
#     #     transition_matrix = load_transition_matrix(path=params['transition_matrix'],
#     #                                                palette=palette)
#     #     # select expected final states
#     #     self.final_palette = transition_matrix.getfinal_palette(info_u=self.state)
        
#     #     # calibration
#     #     try:
#     #         calibration_method = params['calibration_method']
#     #     except:
#     #         calibration_method = DEFAULT_calibration_method
        
#     #     try:
#     #         calibration_params = params['calibration_params']
#     #     except:
#     #         calibration_params = {}
        
#     #     if calibration_method == 'bayes':
#     #         try:
#     #             density_estimation_method = calibration_params['density_estimation_method']
#     #         except:
#     #             density_estimation_method = DEFAULT_calibration_params_density_estimation_method
                
#     #         de_class = _density_estimation_methods[density_estimation_method]
#     #         de_parameters = extract_parameters(de_class, calibration_params)

#     #         tpe = Bayes(density_estimator=de_class(verbose=self.verbose,
#     #                                                **de_parameters),
#     #                     verbose=self.verbose,
#     #                     verbose_heading_level=4)

#     #         for state_v in self.final_palette:
#     #             add_cde_parameters = extract_parameters(tpe.add_conditional_density_estimator, calibration_params)

#     #             cde_class = _density_estimation_methods[density_estimation_method]
#     #             cde_parameters = extract_parameters(cde_class, calibration_params)

#     #             tpe.add_conditional_density_estimator(
#     #                 state=state_v,
#     #                 density_estimator=cde_class(verbose=self.verbose,
#     #                                             # verbose_heading_level=5,
#     #                                             **cde_parameters),
#     #                 **add_cde_parameters)
            
#     #         self.transition_probability_estimator = tpe
        
#         # # allocation
#         # try:
#         #     allocation_method = params['allocation_method']
#         # except:
#         #     allocation_method = DEFAULT_allocation_method
            
#         # try:
#         #     allocation_params = params['allocation_params']
#         # except:
#         #     allocation_params = {}
        
#         # alloc_class = _allocation_methods[allocation_method]
#         # alloc_parameters = extract_parameters(alloc_class, params)

#         # self.allocator = alloc_class(verbose=self.verbose,
#         #                         verbose_heading_level=3,
#         #                         **alloc_parameters)
        
#         # try:
#         #     self.set_features_bounds = params['set_features_bounds']
#         # except:
#         #     self.set_features_bounds = DEFAULT_set_features_bounds
        
#         # try:
#         #     self.feature_selection = params['feature_selection']
#         # except:
#         #     self.feature_selection = DEFAULT_feature_selection
        
#         # try:
#         #     self.fit_bootstrap_patches = params['fit_bootstrap_patches']
#         # except:
#         #     self.fit_bootstrap_patches = DEFAULT_fit_bootstrap_patches

# def _get_n_decimals(s):
#     try:
#         int(s.rstrip('0').rstrip('.'))
#         return 0
#     except:
#         return len(str(float(s)).split('.')[-1])
