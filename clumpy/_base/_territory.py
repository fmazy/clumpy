#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 15:40:56 2021

@author: frem
"""

from ._layer import LandUseLayer, ProbaLayer
from . import Region
from ..tools._path import path_split
from ..tools._console import title_heading
from ._feature import Features

import numpy as np

import logging
logger = logging.getLogger('clumpy')

class Territory():
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
                 regions=None,
                 verbose=0,
                 verbose_heading_level=1):

        self.regions = regions
        if self.regions is None:
            self.regions = {}

        self.verbose = verbose
        self.verbose_heading_level = verbose_heading_level
        
        self.lul = {}
    
    def add_region(self, region):
        """
        Add a region. If ``region`` is already in the list, nothing happened.

        Parameters
        ----------
        region : Region
            The region to append.

        Returns
        -------
        self
        """
        region.territory = self
        if region not in self.regions.values():
            self.regions[region.label] = region
        
        return (self)

    def remove_region(self, region):
        """
        Remove a region.

        Parameters
        ----------
        region : Region
            The region to remove

        Returns
        -------
        self
        """
        try:
            del self.regions[region.label]
        except:
            pass

        return (self)    
    
    def set_lul(self, lul, kind):
        self.lul[kind] = lul
        return(self)
    
    def get_lul(self, kind):
        return(self.lul[kind])

    def check(self, objects=[]):
        """
        Check the Region object through regions checks.
        Notably, estimators uniqueness are checked to avoid malfunctioning during transition probabilities estimation.
        """
        for region in self.regions.values():
            
            if region in objects:
                raise(ValueError("Region objects must be different."))
            else:
                objects.append(region)
            
            region.check(objects=objects)

    def get_region(self, label):
        try:
            return(self.regions[label])
        except:
            logger.error('The given label region does not exist : '+str(label))
            raise

    def make(self, case):
        self.regions = {}
        
        for region_label, region_params in case.params['regions'].items():
            region = Region(label=region_label,
                            verbose=case.get_verbose(),
                            verbose_heading_level=2)
            
            region.make(palette=case.palette, 
                        **region_params)
            
            
            self.add_region(region)

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

        for region in self.regions.values():
            region.fit(distances_to_states=distances_to_states)

        if self.verbose > 0:
            print('Territory fitting done.\n')

        return self

    def transition_matrices(self,
                            lul_initial,
                            lul_final,
                            masks=None):
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
        if masks is None:
            masks = {region: None for region in self.regions.values()}

        tms = {}

        for region in self.regions.values():
            tms[region] = region.transition_matrix(lul_initial=lul_initial,
                                                   lul_final=lul_final,
                                                   mask=masks[region])

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
        
        for region_label, region in self.regions.items():
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
        
        M = np.zeros((0,) + lul.get_data().shape)
        initial_final = []
        
        for region_label, region in self.regions.items():
            M__region, initial_states__region, final_states__region = region._get_transition_probabilities_layer_data(
                lul=lul,
                effective_transitions_only=effective_transitions_only)
            
            for i in range(len(initial_states__region)):
                initial_final__i = (initial_states__region[i],
                                    final_states__region[i])
                if initial_final__i in initial_final:
                    i_band = initial_final.index(initial_final__i)
                    M[i_band] += M__region[i]
                else:
                    M = np.concatenate((M, M__region[[i]]))
                    initial_final.append(initial_final__i)
        
        initial_states = [initial for initial, final in initial_final]
        final_states = [final for initial, final in initial_final]
        
        return(M, initial_states, final_states)
            
            
        

    def allocate(self,
                 regions_transition_matrices,
                 lul,
                 masks=None,
                 path=None,
                 path_prefix_transition_probabilities=None):
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

        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level) + 'Territory allocation\n')

        if masks is None:
            masks = {region: None for region in self.regions.values()}

        # convert keys if label strings
        masks_region_keys = masks.copy()
        for key, mask in masks.items():
            if isinstance(key, str):
                del masks_region_keys[key]
                masks_region_keys[self.get_region(key)] = mask

        masks = masks_region_keys

        # same for transition_matrices
        tms_copy = regions_transition_matrices.copy()
        for key, tm in regions_transition_matrices.items():
            if isinstance(key, str):
                del tms_copy[key]
                tms_copy[self.get_region(key)] = tm

        regions_transition_matrices = tms_copy

        distances_to_states = {}

        lul_data = lul.get_data().copy()

        for region in self.regions.values():

            if path_prefix_transition_probabilities is not None:
                region_path_prefix_transition_probabilities = path_prefix_transition_probabilities + '_' + str(
                    region.label)
            else:
                region_path_prefix_transition_probabilities = None

            region.allocate(transition_matrix=regions_transition_matrices[region],
                            lul=lul_data,
                            lul_origin=lul,
                            mask=masks[region],
                            distances_to_states=distances_to_states,
                            path=None,
                            path_prefix_transition_probabilities=region_path_prefix_transition_probabilities,
                            copy_geo=lul)
            # Note that the path is set to None above in order to allocate through all regions and save in a second time !

        if path is not None:
            folder_path, file_name, file_ext = path_split(path)
            return (LandUseLayer(label=file_name,
                                 data=lul_data,
                                 copy_geo=lul,
                                 path=path,
                                 palette=lul.palette))

        if self.verbose > 0:
            print('Territory allocate done.\n')

    def iterative_allocate(self,
                           n_iter,
                           transition_matrices,
                           lul,
                           masks=None,
                           path_prefix=None,
                           path_prefix_transition_probabilities=None):
        """
        Multisteps allocation

        Parameters
        ----------
        n_iter : int
            Number of iterations to allocate.

        transition_matrices : dict(Region:TransitionMatrix)
            Dict of transition matrix with the corresponding region as key.

        lul : LandUseLayer or ndarray
            The studied land use layer.

        masks : dict(Region:MaskLayer), default=None
            Dict of masks layer with the corresponding region as key. If None, the whole map is used for each region.

        path_prefix : str or callable, default=None
            The path to save every allocated step map as ``path_prefix+'_iter'+str(i)+'.tif'``.
            If callable, it has to be a function with the iteration ``i`` in entry
            which returns the tif file path as a string.
            If None, the allocation is only saved within `lul`, if `lul` is a ndarray.
            Note that if ``path`` is not ``None``, ``lul`` must be LandUseLayer.

        path_prefix_transition_probabilities : str or callable, default=None
            The patch to save every transition probabilities step map
            as ``path_prefix_transition_probabilities+'_iter'+str(i)+'_'+region.label+'_'+u+'_'+v+'.tif'.``
            If callable, it has to be a function with the iteration ``i`` in entry
            which returns the path prefix as a string.

        Returns
        -------
        lul_allocated : LandUseLayer
            The allocated map as a land use layer.
        """

        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level) + 'Territory iterative allocation\n')

        lul_step = lul

        self.verbose_heading_level += 1
        for i in range(1,n_iter+1):

            if isinstance(path_prefix, str):
                iter_path = path_prefix + '_iter' + str(i) + '.tif'
            elif callable(path_prefix):
                iter_path = path_prefix(i)
            else:
                iter_path = None

            if isinstance(path_prefix_transition_probabilities, str):
                iter_path_prefix_transition_probabilities = path_prefix_transition_probabilities + '_iter' + str(i) + '.tif'
            elif callable(path_prefix_transition_probabilities):
                iter_path_prefix_transition_probabilities = path_prefix_transition_probabilities(i)
            else:
                iter_path_prefix_transition_probabilities = None

            lul_step = self.allocate(transition_matrices=transition_matrices,
                                     lul=lul_step,
                                     masks=masks,
                                     path=iter_path,
                                     path_prefix_transition_probabilities=iter_path_prefix_transition_probabilities)

        self.verbose_heading_level -= 1

        if path_prefix is not None:
            return lul_step
