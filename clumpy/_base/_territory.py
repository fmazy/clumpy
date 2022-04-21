#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 15:40:56 2021

@author: frem
"""

from ..layer import LandUseLayer, ProbaLayer, create_proba_layer
from . import Region
from ..tools._path import path_split
from ..tools._console import title_heading

import numpy as np

import logging
logger = logging.getLogger('clumpy')

class Territory(dict):
    """
    Territory.

    Parameters
    ----------
    regions : list(Region)
        List of regions which constitute the territory. If ``None``, an empty list is created
        and can be append by ``self.add_region()``.
    verbose : int, default=0
        Verbosity level.

    verbose_heading_level : int, default=1
        Verbose heading level for markdown titles. If ``0``, no markdown title are printed.
    """

    def __init__(self,
                 verbose=0,
                 verbose_heading_level=1):
                
        self.features = None
        
        self.verbose = verbose
        self.verbose_heading_level = verbose_heading_level
        
        self.lul = {}
    
    def add_region(self,
                   region):
        self[region.label] = region
        region.territory = self
        return(self)
    
    def set_lul(self, lul, kind):
        self.lul[kind] = lul
        return(self)
    
    def get_lul(self, kind):
        return(self.lul[kind])
    
    def set_features(self, features):
        self.features = features
        return(self)
    
    def get_features(self):
        return(self.features)

    def check(self, objects=None):
        """
        Check the unicity of objects.
        Notably, estimators uniqueness are checked to avoid malfunctioning during transition probabilities estimation.
        """
        if objects is None:
            objects = []
        
        for region in self.values():
            if region in objects:
                raise(ValueError("Region objects must be different."))
            else:
                objects.append(region)
            
            region.check(objects=objects)
        
    def fit(self):
        """
        Fit the territory.

        Parameters
        ----------
        lul_initial : LandUseLayer
            The initial land use layer.

        lul_final : LandUseLayer
            The final land use layer.

        masks : dict(Region:MaskLayer), default=None
            Dict of masks layer with the corresponding region as key. If None, the whole map is used for each region.

        Returns
        -------
        self
        """

        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level) + 'Territory fitting\n')

        # convert keys if label strings
        distances_to_states = {}

        for region in self.values():
            region.fit(distances_to_states=distances_to_states)

        if self.verbose > 0:
            print('Territory fitting done.\n')

        return self

    def compute_transition_matrix(self,
                                  lul_initial=None,
                                  lul_final=None,
                                  mask=None,
                                  final_states_only=True):
        """
        Compute transition matrices

        Parameters
        ----------
        lul_initial : LandUseLayer
            The initial land use.

        lul_final : LandUseLayer
            The final land use.

        mask : MaskLayer, default = None
            The region mask layer. If ``None``, the whole area is studied.

        Returns
        -------
        tms : dict(Region:TransitionMatrix)
            A dict of transition matrices with regions as keys.
        """
        
        if lul_initial is None:
            lul_initial = 'initial'
        if isinstance(lul_initial, str):
            lul_initial = self.get_lul(lul_initial)
        
        if lul_final is None:
            lul_final = 'final'
        if isinstance(lul_final, str):
            lul_final = self.get_lul(lul_final)
                
        tms = {}

        for region_label, region in self.items():
            tms[region_label] = region.compute_transition_matrix(lul_initial=lul_initial,
                                                                 lul_final=lul_final,
                                                                 mask=mask,
                                                                 final_states_only=final_states_only)

        return (tms)

    def transition_probabilities(self,
                                 lul,
                                 effective_transitions_only=True):
        """
        Compute transition probabilities.

        Parameters
        ----------

        lul : LandUseLayer
            The studied land use layer.

        Returns
        -------
        tp : dict(Region:[J,P_v__u_Y])
            Dict of regional results with corresponding region as key. Th regional results are ``J``, a ndarray of shape (n_samples,)
            which is the element indices in the flattened map and ``P_v__u_Y``, the transition probabilities whose columns correspond
            to transition matrix argument : ``palette_v``.
        """
        if isinstance(lul, str):
            lul = self.get_lul(lul)
        
        r = {}
        
        for region_label, region in self.items():
            r[region_label] = region.transition_probabilities(
                lul=lul,
                effective_transitions_only=effective_transitions_only)
        
        return r

    def transition_probabilities_layer(self,
                                       lul,
                                       path,
                                       effective_transitions_only=True):
        
        if isinstance(lul, str):
            lul = self.get_lul(lul)
        
        p = self.transition_probabilities(
            lul=lul,
            effective_transitions_only=effective_transitions_only)
        
        proba_layer = create_proba_layer(path=path,
                                         lul=lul,
                                         p=p)
        
        return(proba_layer)
            
    def allocate(self,
                 p,
                 lul,
                 lul_origin=None,
                 distances_to_states={}):
        """
        Allocate.

        Parameters
        ----------
        regions_transition_matrices : dict(Region:TransitionMatrix)
            Dict of transition matrix with the corresponding region as key.

        lul : LandUseLayer
            The studied land use layer.

        masks : dict(Region:MaskLayer), default=None
            Dict of masks layer with the corresponding region as key. If None, the whole map is used for each region.

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

        if lul is None:
            lul = 'start'
        
        if isinstance(lul, str):
            lul = self.get_lul(lul)
        
        if isinstance(lul, LandUseLayer):
            lul_data = lul.get_data()
        else:
            lul_data = lul
        
        if lul_origin is None:
            lul_origin_data = lul_data.copy()
        else:
            lul_origin_data = lul_origin
        
        for region_label, p_region in p.items():
            self[region_label].allocate(p=p_region,
                                        lul=lul_data,
                                        lul_origin=lul_origin_data,
                                        distances_to_states=distances_to_states)
        
        return(lul_data)
    
    def allocate_layer(self,
                       path,
                       p,
                       lul,
                       lul_origin=None,
                       distances_to_states={}):
        lul_data = self.allocate(p=p,
                                 lul=lul,
                                 lul_origin=lul_origin,
                                 distances_to_states=distances_to_states)
        
        alloc_layer = LandUseLayer(path=path,
                                   data=lul_data,
                                   copy_geo=lul,
                                   palette=lul.palette)
        return(alloc_layer)

    # def iterative_allocate(self,
    #                        n_iter,
    #                        transition_matrices,
    #                        lul,
    #                        masks=None,
    #                        path_prefix=None,
    #                        path_prefix_transition_probabilities=None):
    #     """
    #     Multisteps allocation

    #     Parameters
    #     ----------
    #     n_iter : int
    #         Number of iterations to allocate.

    #     transition_matrices : dict(Region:TransitionMatrix)
    #         Dict of transition matrix with the corresponding region as key.

    #     lul : LandUseLayer or ndarray
    #         The studied land use layer.

    #     masks : dict(Region:MaskLayer), default=None
    #         Dict of masks layer with the corresponding region as key. If None, the whole map is used for each region.

    #     path_prefix : str or callable, default=None
    #         The path to save every allocated step map as ``path_prefix+'_iter'+str(i)+'.tif'``.
    #         If callable, it has to be a function with the iteration ``i`` in entry
    #         which returns the tif file path as a string.
    #         If None, the allocation is only saved within `lul`, if `lul` is a ndarray.
    #         Note that if ``path`` is not ``None``, ``lul`` must be LandUseLayer.

    #     path_prefix_transition_probabilities : str or callable, default=None
    #         The patch to save every transition probabilities step map
    #         as ``path_prefix_transition_probabilities+'_iter'+str(i)+'_'+region.label+'_'+u+'_'+v+'.tif'.``
    #         If callable, it has to be a function with the iteration ``i`` in entry
    #         which returns the path prefix as a string.

    #     Returns
    #     -------
    #     lul_allocated : LandUseLayer
    #         The allocated map as a land use layer.
    #     """

    #     if self.verbose > 0:
    #         print(title_heading(self.verbose_heading_level) + 'Territory iterative allocation\n')

    #     lul_step = lul

    #     self.verbose_heading_level += 1
    #     for i in range(1,n_iter+1):

    #         if isinstance(path_prefix, str):
    #             iter_path = path_prefix + '_iter' + str(i) + '.tif'
    #         elif callable(path_prefix):
    #             iter_path = path_prefix(i)
    #         else:
    #             iter_path = None

    #         if isinstance(path_prefix_transition_probabilities, str):
    #             iter_path_prefix_transition_probabilities = path_prefix_transition_probabilities + '_iter' + str(i) + '.tif'
    #         elif callable(path_prefix_transition_probabilities):
    #             iter_path_prefix_transition_probabilities = path_prefix_transition_probabilities(i)
    #         else:
    #             iter_path_prefix_transition_probabilities = None

    #         lul_step = self.allocate(transition_matrices=transition_matrices,
    #                                  lul=lul_step,
    #                                  masks=masks,
    #                                  path=iter_path,
    #                                  path_prefix_transition_probabilities=iter_path_prefix_transition_probabilities)

    #     self.verbose_heading_level -= 1

    #     if path_prefix is not None:
    #         return lul_step
        
    # def make(self, case):
    #     self = {}
        
    #     for region_label, region_params in case.params['regions'].items():
    #         region = Region(label=region_label,
    #                         verbose=case.get_verbose(),
    #                         verbose_heading_level=2)
            
    #         region.make(palette=case.palette, 
    #                     **region_params)
            
            
    #         self.add_region(region)
