#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from time import time

from scipy import ndimage

# base import
from ._layer import Layer, FeatureLayer, LandUseLayer, MaskLayer, ProbaLayer
from ._state import State
from ._transition_matrix import TransitionMatrix, load_transition_matrix

# Transition Probability Estimator
from ..density_estimation import _methods as _density_estimation_methods
from ..transition_probability_estimation._tpe import TransitionProbabilityEstimator
from ..transition_probability_estimation import Bayes

# features
from ..feature_selection import MRMR
from ._feature import Features

# Tools
from ..tools._path import path_split
from ..tools._console import title_heading
from ..tools._funcs import extract_parameters

# Allocation
from ..allocation._allocator import Allocator
from ..allocation._compute_patches import compute_bootstrap_patches
from ..allocation import _methods as _allocation_methods

import logging
logger = logging.getLogger('clumpy')

DEFAULT_calibration_method = 'bayes'
DEFAULT_calibration_params_density_estimation_method = 'kde'
DEFAULT_allocation_method = 'unbiased'
DEFAULT_set_features_bounds = True
DEFAULT_fit_bootstrap_patches = True

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
                 final_palette=None,
                 transition_probability_estimator=None,
                 set_features_bounds=DEFAULT_set_features_bounds,
                 fit_bootstrap_patches=DEFAULT_fit_bootstrap_patches,
                 allocator=None,
                 verbose=0,
                 verbose_heading_level=1):

        # Transition probability estimator
        if transition_probability_estimator is not None and isinstance(transition_probability_estimator,
                                                                       TransitionProbabilityEstimator) == False:
            raise (TypeError(
                "Unexpected 'transition_probability_estimator. A 'TransitionProbabilityEstimator' object is expected."))
        self.transition_probability_estimator = transition_probability_estimator

        self.calibrator = None
        self.transition_matrix = None
        self.lul = {}
        self.mask = {}
        
        # set features bounds 
        self.set_features_bounds = set_features_bounds
        
        # fit bootstrap patches
        self.fit_bootstrap_patches = fit_bootstrap_patches

        # allocator
        self.allocator = allocator
        
        # state
        self.state = state
        
        self.verbose = verbose
        self.verbose_heading_level = verbose_heading_level
        
        self.region = None
        self.final_palette = final_palette

    def __repr__(self):
        return 'land'

    def set_params(self,
                   **params):
        for key, param in params.items():
            setattr(self, key, param)
    
    def make(self, palette, **params):
        # features
        features = []
        if 'features' in params.keys():
            for feature_params in params['features']:
                if feature_params['type'] == 'layer':
                    fp = extract_parameters(FeatureLayer, feature_params)
                    features.append(FeatureLayer(**fp))
                elif feature_params['type'] == 'distance':
                    if feature_params['state'] != self.state.value:
                        features.append(palette._get_by_value(feature_params['state']))
        
        self.features = features
        
        # feature selection
        if 'feature_selection' in params.keys():
            if isinstance(params['feature_selection'], int):
                self.feature_selection = params['feature_selection']
            else:
                self.feature_selection = -1
        
        # transition matrix
        transition_matrix = load_transition_matrix(path=params['transition_matrix'],
                                                   palette=palette)
        # select expected final states
        self.final_palette = transition_matrix.getfinal_palette(info_u=self.state)
        
        # calibration
        try:
            calibration_method = params['calibration_method']
        except:
            calibration_method = DEFAULT_calibration_method
        
        try:
            calibration_params = params['calibration_params']
        except:
            calibration_params = {}
        
        if calibration_method == 'bayes':
            try:
                density_estimation_method = calibration_params['density_estimation_method']
            except:
                density_estimation_method = DEFAULT_calibration_params_density_estimation_method
                
            de_class = _density_estimation_methods[density_estimation_method]
            de_parameters = extract_parameters(de_class, calibration_params)

            tpe = Bayes(density_estimator=de_class(verbose=self.verbose,
                                                   **de_parameters),
                        verbose=self.verbose,
                        verbose_heading_level=4)

            for state_v in self.final_palette:
                add_cde_parameters = extract_parameters(tpe.add_conditional_density_estimator, calibration_params)

                cde_class = _density_estimation_methods[density_estimation_method]
                cde_parameters = extract_parameters(cde_class, calibration_params)

                tpe.add_conditional_density_estimator(
                    state=state_v,
                    density_estimator=cde_class(verbose=self.verbose,
                                                # verbose_heading_level=5,
                                                **cde_parameters),
                    **add_cde_parameters)
            
            self.transition_probability_estimator = tpe
        
        # allocation
        try:
            allocation_method = params['allocation_method']
        except:
            allocation_method = DEFAULT_allocation_method
            
        try:
            allocation_params = params['allocation_params']
        except:
            allocation_params = {}
        
        alloc_class = _allocation_methods[allocation_method]
        alloc_parameters = extract_parameters(alloc_class, params)

        self.allocator = alloc_class(verbose=self.verbose,
                                verbose_heading_level=3,
                                **alloc_parameters)
        
        try:
            self.set_features_bounds = params['set_features_bounds']
        except:
            self.set_features_bounds = DEFAULT_set_features_bounds
        
        try:
            self.feature_selection = params['feature_selection']
        except:
            self.feature_selection = DEFAULT_feature_selection
        
        try:
            self.fit_bootstrap_patches = params['fit_bootstrap_patches']
        except:
            self.fit_bootstrap_patches = DEFAULT_fit_bootstrap_patches

    def check(self, objects=[]):
        """
        Check the Land object.
        Notably, estimators uniqueness are checked to avoid malfunctioning during transition probabilities estimation.
        """
        if self.calibrator in objects:
            raise(ValueError("Calibrator objects must be different."))
        else:
            objects.append(self.calibrator)
        
        self.calibrator.check(objects=objects)

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

        if isinstance(self.feature_selector, list):
            feature_selector = self.feature_selector
        else:
            feature_selector = [self.feature_selector]

        for fs in feature_selector:
            if fs in feature_selectors and fs is not None:
                raise (ValueError('The feature selection is already used. A new featureSelector must be invoked.'))
            feature_selectors.append(fs)

        return feature_selectors
        
    def set_calibrator(self, calibrator):
        self.calibrator = calibrator
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
    
    def get_J(self, 
              lul,
              mask=None):
        """
        Get J indices.

        Parameters
        ----------
        lul : {'initial', 'final', 'start'} or LandUseLayer or np.array
            The land use map.
        mask : {'calibration', 'allocation'} or MaskLayer or np.array
            The mask.

        Returns
        -------
        None.

        """
        
        if isinstance(lul, str):
            lul = self.get_lul(lul)
        
        if isinstance(mask, str):
            mask = self.get_mask(mask)
                
        # initial data
        # the region is selected after the distance computation
        if isinstance(lul, LandUseLayer):
            data_lul = lul.get_data().copy()
        else:
            data_lul = lul.copy()

        # selection according to the region.
        # one set -1 to non studied data
        # -1 is a forbiden state value.
        if mask is not None:
            if isinstance(mask, MaskLayer):
                data_lul[mask.get_data() != 1] = -1
            else:
                data_lul[mask != 1] = -1

        # get pixels indexes whose initial states are u
        return(np.where(data_lul.flat == int(self.state))[0])
    
    def get_V(self, 
              lul,
              J,
              restrict_to_final_palette=True):
        
        if isinstance(lul, str):
            lul = self.get_lul(lul)
        
        if isinstance(lul, LandUseLayer):
            data_lul = lul.get_data().copy()
        else:
            data_lul = lul.copy()
        
        V = data_lul.flat[J]
            
        if restrict_to_final_palette:
            tm = self.region.get_transition_matrix()
            final_palette = tm.get_final_palette(self.state)
            V[~np.isin(V, final_palette.get_list_of_values())] = int(self.state)
        
        return(V)
    
    def get_J_V(self,
                lul_initial,
                lul_final,
                mask=None,
                restrict_to_final_palette=True):
        J = self.get_J(lul=lul_initial,
                       mask=mask)
        V = self.get_V(lul=lul_final,
                       J=J,
                       restrict_to_final_palette=restrict_to_final_palette)
        return(J, V)

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
        self._time_fit = {}
        st0 = time()
        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level) + 'Land ' + str(self.state) + ' fitting\n')
        
        J, V = self.get_J_V(lul_initial='initial',
                            lul_final='final',
                            mask='calibration',
                            restrict_to_final_palette=True)
                
        self.calibrator.fit(J=J,
                            V=V,
                            state=self.state,
                            lul=self.get_lul('initial'),
                            distances_to_states=distances_to_states)
        
        # if self.transition_probability_estimator is None:
        #     raise (ValueError('Transition probability estimator is expected for fitting.'))

        # self._fit_tpe(lul_initial=lul_initial,
        #               lul_final=lul_final,
        #               mask=mask,
        #               distances_to_states=distances_to_states)

        # if self.fit_bootstrap_patches:
        #     st = time()
        #     self.compute_bootstrap_patches(
        #         palette_v=self.transition_probability_estimator._palette_fitted_states,
        #         lul_initial=lul_initial,
        #         lul_final=lul_final,
        #         mask=mask)
        #     self._time_fit['compute_bootstrap_patches'] = time()-st

        if self.verbose > 0:
            print('Land ' + str(self.state) + ' fitting done.\n')

        self._time_fit['all'] = time()-st0

        return self
    
    def transition_probabilities(self, 
                                 lul='start',
                                 effective_transitions_only=True):
        if isinstance(lul, str):
            lul = self.get_lul(lul)
            
        J = self.get_J(lul=lul,
                       mask='allocation')
        
        tm = self.get_transition_matrix().extract(self.state)
    
        P_v__u_Y = self.calibrator.transition_probabilities(J=J,
                                                            tm=tm,
                                                            lul=lul,
                                                            distances_to_states={})
        
        final_states = tm.palette_v.get_list_of_values()
        
        if effective_transitions_only:
            bands_to_keep = np.array(final_states) != int(self.state)
            final_states = list(np.array(final_states)[bands_to_keep])
            P_v__u_Y = P_v__u_Y[:, bands_to_keep]
        
        return(J, P_v__u_Y, final_states)
    
    def transition_probabilities_layer(self, 
                                       path,
                                       lul='start',
                                       effective_transitions_only=True):
        
        if isinstance(lul, str):
            lul = self.get_lul(lul)
        
        M, initial_states, final_states = self._get_transition_probabilities_layer_data(
            lul=lul,
            effective_transitions_only=effective_transitions_only)
                
        probalayer = ProbaLayer(path=path,
                                data=M,
                                initial_states = initial_states,
                                final_states = final_states,
                                copy_geo=lul)
        
        return(probalayer)
            
    def _get_transition_probabilities_layer_data(self, 
                                                 lul='start',
                                                 effective_transitions_only=True):
        
        if isinstance(lul, str):
            lul = self.get_lul(lul)
        
        shape = lul.get_data().shape
        
        J, P_v__u_Y, final_states = self.transition_probabilities(
            lul=lul,
            effective_transitions_only=effective_transitions_only)
                
        n_bands = len(final_states)
        M = np.zeros((n_bands,) + shape)
        
        for i_band in range(n_bands):
            M[i_band].flat[J] = P_v__u_Y[:, i_band]
        
        initial_states = [int(self.state) for i in range(len(final_states))]
        
        return(M, initial_states, final_states)


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

    def compute_bootstrap_patches(self,
                                  palette_v,
                                  lul_initial,
                                  lul_final,
                                  mask):
        """
        Compute Bootstrap patches

        """
        patches = compute_bootstrap_patches(state=self.state,
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


def _compute_distance(state, data, distances_to_states):
    v_matrix = (data == state.value).astype(int)
    distances_to_states[state] = ndimage.distance_transform_edt(1 - v_matrix)


def _get_n_decimals(s):
    try:
        int(s.rstrip('0').rstrip('.'))
        return 0
    except:
        return len(str(float(s)).split('.')[-1])
