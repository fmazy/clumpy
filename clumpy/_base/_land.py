#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from time import time

from scipy import ndimage

# base import
from ..layer import Layer, FeatureLayer, LandUseLayer, MaskLayer, ProbaLayer, create_proba_layer
from ._state import State
from ._transition_matrix import TransitionMatrix, load_transition_matrix

# Transition Probability Estimator
from ..density_estimation import _methods as _density_estimation_methods
from ..transition_probability_estimation._tpe import TransitionProbabilityEstimator
from ..transition_probability_estimation import Bayes

# features
from ..feature_selection import MRMR

# Tools
from ..tools._path import path_split
from ..tools._console import title_heading
from ..tools._funcs import extract_parameters

# Allocation
from ..allocation._allocator import Allocator
from ..allocation import _methods as _allocation_methods

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
                 calibrator=None,
                 allocator=None,
                 verbose=0,
                 verbose_heading_level=1):
        
        # state
        self.state = state
        
        self.calibrator = calibrator
        self.allocator = allocator
        
        self.verbose = verbose
        self.verbose_heading_level = verbose_heading_level
        
        self.region = None
        self.features = None
        self.transition_matrix = None
        self.lul = {}
        self.mask = {}

    def __repr__(self):
        return 'Land()'

    def set_params(self,
                   **params):
        for key, param in params.items():
            setattr(self, key, param)

    def check(self, objects=None):
        """
        Check the unicity of objects.
        Notably, estimators uniqueness are checked to avoid malfunctioning during transition probabilities estimation.
        """
        if objects is None:
            objects = []
            
        if self.calibrator in objects:
            raise(ValueError("Calibrator objects must be different."))
        else:
            objects.append(self.calibrator)
        
        self.calibrator.check(objects=objects)
        
    def set_calibrator(self, calibrator):
        self.calibrator = calibrator
        return(self)
    
    def set_allocator(self, allocator):
        self.allocator = allocator
        return(self)    
       
    def set_lul(self, lul, kind):
        self.lul[kind] = lul
        return(self)
    
    def get_lul(self, kind):
        if kind not in self.lul.keys():
            return(self.region.get_lul(kind))
        else:
            return(self.lul[kind])
        
    def set_mask(self, mask, kind):
        self.mask[kind] = mask
        return(self)
    
    def get_mask(self, kind):
        if kind not in self.mask.keys():
            return(self.region.get_mask(kind))
        else:
            return(self.mask[kind])
    
    def set_transition_matrix(self, tm):
        self.transition_matrix = tm
        return(self)
    
    def get_transition_matrix(self):
        if self.transition_matrix is None:
            return(self.region.get_transition_matrix())
        else:
            return(self.transition_matrix)
    
    def set_features(self, features):
        self.features = features
        return(self)
    
    def get_features(self):
        if self.features is None:
            return(self.region.get_features())
        else:
            return(self.features)

    def fit(self,
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
            print(title_heading(self.verbose_heading_level) + 'Land ' + str(self.state) + ' fitting\n')
        
        self.calibrator.fit(lul_initial=self.get_lul('initial'),
               lul_final=self.get_lul('final'),
               features=self.get_features(),
               mask=self.get_mask('calibration'),
               distances_to_states=distances_to_states)
                        
        if self.verbose > 0:
            print('Land ' + str(self.state) + ' fitting done.\n')

        return self
    
    def transition_probabilities(self, 
                                 lul=None,
                                 mask=None,
                                 effective_transitions_only=True,
                                 territory_format=False):
        if lul is None:
            lul = 'start'
        
        if isinstance(lul, str):
            lul = self.get_lul(lul)
        
        if mask is None:
            mask = self.get_mask('allocation')
        
        tm = self.get_transition_matrix().extract(self.state)
    
        p = self.calibrator.transition_probabilities(
            lul=lul,
            tm=tm,
            features=self.get_features(),
            mask = mask,
            distances_to_states={},
            effective_transitions_only=effective_transitions_only)
        
        if not territory_format:
            return(p)
        else:
            return({self.region.label : {int(self.state) : p}})
    
    def transition_probabilities_layer(self, 
                                       path,
                                       lul='start',
                                       effective_transitions_only=True):
        
        if isinstance(lul, str):
            lul = self.get_lul(lul)
        
        p = self.transition_probabilities(
            lul=lul,
            effective_transitions_only=effective_transitions_only,
            territory_format=True)
        
        proba_layer = create_proba_layer(path=path,
                                         lul=lul,
                                         p=p)
        
        return(proba_layer)
            
    def transition_matrix(self):
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
        
        lul_initial = self.get_lul('initial')
        lul_final = self.get_lul('final')
        mask = self.get_mask('calibration')
        
        J, V = self.get_values(kind='calibration',
                               explanatory_variables=False)

        v_unique, n_counts = np.unique(V, return_counts=True)
        P_v = n_counts / n_counts.sum()
        P_v = P_v[None, :]

        v_unique = v_unique.astype(int)

        palette_u = lul_initial.palette.extract(infos=[self.state])
        palette_v = lul_final.palette.extract(infos=v_unique)

        return (TransitionMatrix(M=P_v,
                                 palette_u=palette_u,
                                 palette_v=palette_v))

    

    def allocate(self,
                 transition_matrix,
                 lul,
                 lul_origin=None,
                 mask=None,
                 distances_to_states={},
                 path=None,
                 path_prefix_transition_probabilities=None,
                 copy_geo=None):
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

        path_prefix_transition_probabilities : str, default=None
            The path prefix to save transition probabilities.

        Returns
        -------
        lul_allocated : LandUseLayer
            Only returned if ``path`` is not ``None``. The allocated map as a land use layer.
        """
        # check if it is really a land transition matrix
        transition_matrix._check_land_transition_matrix()

        state = transition_matrix.palette_u.states[0]
        
        if not isinstance(self.allocator, Allocator):
            raise (ValueError("Unexpected 'allocator'. A clumpy.allocation.Allocator object is expected ; got instead "+str(type(self.allocator))))

        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level) + 'Land ' + str(state) + ' allocation\n')

        self.allocator.allocate(transition_matrix=transition_matrix,
                                land=self,
                                lul=lul,
                                lul_origin=lul_origin,
                                mask=mask,
                                distances_to_states=distances_to_states,
                                path=path,
                                path_prefix_transition_probabilities=path_prefix_transition_probabilities,
                                copy_geo=copy_geo)

        if self.verbose > 0:
            print('Land ' + str(state) + ' allocation done.\n')

    def dinamica_determine_ranges(self,
                                  lul_initial,
                                  params,
                                  mask=None):
        J, X = self.get_values(lul_initial=lul_initial,
                               mask=mask,
                               explanatory_variables=True)

        ranges = {}
        delta = {}

        for id_feature, feature in enumerate(self.features):
            param = params[feature]

            x = X[:, id_feature].copy()
            n_round = _get_n_decimals(param['increment'])
            x = np.sort(x)
            x = np.round(x, n_round)

            ranges[feature] = [np.round(x[0], n_round)]
            delta[feature] = [0, 0]

            for i, xi in enumerate(x):
                if delta[feature][-1] >= param['maximum_delta']:
                    ranges[feature].append(xi)
                    delta[feature].append(1)
                elif xi - ranges[feature][-1] > param['increment'] and delta[feature][-1] >= param['minimum_delta']:
                    ranges[feature].append(ranges[feature][-1] + param['increment'])
                    delta[feature].append(1)

                elif len(ranges[feature]) > 1:
                    v1 = np.array([ranges[feature][-1] - ranges[feature][-2],
                                   (delta[feature][-2] - delta[feature][-3])])
                    v2 = np.array([xi - ranges[feature][-1],
                                   (delta[feature][-1] + 1 - delta[feature][-2])])

                    norm_v1 = np.linalg.norm(v1)
                    norm_v2 = np.linalg.norm(v2)
                    if norm_v1 > 0 and norm_v2 > 0:
                        v1 /= norm_v1
                        v2 /= norm_v2

                        dot = v1[0] * v2[0] + v1[1] * v2[1]
                        if dot >= 0 and dot <= 1:
                            angle = np.arccos(np.abs(v1[0] * v2[0] + v1[1] * v2[1])) * 180 / np.pi
                        else:
                            angle = 0
                    else:
                        angle = 0

                    if angle > param['tolerance_angle'] and delta[feature][-1] >= param['minimum_delta']:
                        ranges[feature].append(xi)
                        delta[feature].append(1)
                    else:
                        delta[feature][-1] += 1
                else:
                    delta[feature][-1] += 1

        return (ranges, delta)
    
# def make(self, palette, **params):
    #     # features
    #     features = []
    #     if 'features' in params.keys():
    #         for feature_params in params['features']:
    #             if feature_params['type'] == 'layer':
    #                 fp = extract_parameters(FeatureLayer, feature_params)
    #                 features.append(FeatureLayer(**fp))
    #             elif feature_params['type'] == 'distance':
    #                 if feature_params['state'] != self.state.value:
    #                     features.append(palette._get_by_value(feature_params['state']))
        
    #     self.features = features
        
    #     # feature selection
    #     if 'feature_selection' in params.keys():
    #         if isinstance(params['feature_selection'], int):
    #             self.feature_selection = params['feature_selection']
    #         else:
    #             self.feature_selection = -1
        
    #     # transition matrix
    #     transition_matrix = load_transition_matrix(path=params['transition_matrix'],
    #                                                palette=palette)
    #     # select expected final states
    #     self.final_palette = transition_matrix.getfinal_palette(info_u=self.state)
        
    #     # calibration
    #     try:
    #         calibration_method = params['calibration_method']
    #     except:
    #         calibration_method = DEFAULT_calibration_method
        
    #     try:
    #         calibration_params = params['calibration_params']
    #     except:
    #         calibration_params = {}
        
    #     if calibration_method == 'bayes':
    #         try:
    #             density_estimation_method = calibration_params['density_estimation_method']
    #         except:
    #             density_estimation_method = DEFAULT_calibration_params_density_estimation_method
                
    #         de_class = _density_estimation_methods[density_estimation_method]
    #         de_parameters = extract_parameters(de_class, calibration_params)

    #         tpe = Bayes(density_estimator=de_class(verbose=self.verbose,
    #                                                **de_parameters),
    #                     verbose=self.verbose,
    #                     verbose_heading_level=4)

    #         for state_v in self.final_palette:
    #             add_cde_parameters = extract_parameters(tpe.add_conditional_density_estimator, calibration_params)

    #             cde_class = _density_estimation_methods[density_estimation_method]
    #             cde_parameters = extract_parameters(cde_class, calibration_params)

    #             tpe.add_conditional_density_estimator(
    #                 state=state_v,
    #                 density_estimator=cde_class(verbose=self.verbose,
    #                                             # verbose_heading_level=5,
    #                                             **cde_parameters),
    #                 **add_cde_parameters)
            
    #         self.transition_probability_estimator = tpe
        
        # # allocation
        # try:
        #     allocation_method = params['allocation_method']
        # except:
        #     allocation_method = DEFAULT_allocation_method
            
        # try:
        #     allocation_params = params['allocation_params']
        # except:
        #     allocation_params = {}
        
        # alloc_class = _allocation_methods[allocation_method]
        # alloc_parameters = extract_parameters(alloc_class, params)

        # self.allocator = alloc_class(verbose=self.verbose,
        #                         verbose_heading_level=3,
        #                         **alloc_parameters)
        
        # try:
        #     self.set_features_bounds = params['set_features_bounds']
        # except:
        #     self.set_features_bounds = DEFAULT_set_features_bounds
        
        # try:
        #     self.feature_selection = params['feature_selection']
        # except:
        #     self.feature_selection = DEFAULT_feature_selection
        
        # try:
        #     self.fit_bootstrap_patches = params['fit_bootstrap_patches']
        # except:
        #     self.fit_bootstrap_patches = DEFAULT_fit_bootstrap_patches

def _get_n_decimals(s):
    try:
        int(s.rstrip('0').rstrip('.'))
        return 0
    except:
        return len(str(float(s)).split('.')[-1])
