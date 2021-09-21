#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 15:40:56 2021

@author: frem
"""

from . import LandUseLayer
from ..tools._path import path_split

# TODO : ajouter un self.verbosity_head_level à toutes les classes pour connaître le niveau de titre markdown.

class Territory():
    """
    Territory.

    Parameters
    ----------
    regions : list(Region)
        List of regions which constitute the territory. If ``None``, an empty list is created
        and can be append by ``self.add_region()``.
    """

    def __init__(self,
                 regions=None):

        self.regions = regions
        if self.regions is None:
            self.regions = []

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
        if region not in self.regions:
            self.regions.append(region)

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
        self.regions.remove(region)

        return (self)

    def check(self):
        """
        Check the Region object through regions checks.
        Notably, estimators uniqueness are checked to avoid malfunctioning during transition probabilities estimation.
        """
        density_estimators = []
        feature_selectors = []
        for region in self.regions:
            density_estimators = region._check_density_estimators(density_estimators=density_estimators)
            feature_selectors = region._check_feature_selectors(feature_selectors=feature_selectors)

    def fit(self,
            lul_initial,
            lul_final,
            masks=None):
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
            print('\n## Territory fitting\n')

        if masks is None:
            masks = {region: None for region in self.regions}

        distances_to_states = {}

        for id_region, region in enumerate(self.regions):
            region.fit(lul_initial=lul_initial,
                       lul_final=lul_final,
                       mask=masks[region],
                       distances_to_states=distances_to_states)

        if self.verbose > 0:
            print('Territory fitting done.\n')

        return (self)

    def transition_probabilities(self,
                                 transition_matrices,
                                 lul,
                                 masks=None,
                                 path_prefix=None):
        """
        Compute transition probabilities.

        Parameters
        ----------
        transition_matrices : dict(Region:TransitionMatrix)
            Dict of transition matrix with the corresponding region as key.

        lul : LandUseLayer
            The studied land use layer.

        masks : dict(Region:MaskLayer), default=None
            Dict of masks layer with the corresponding region as key. If None, the whole map is used for each region.

        path_prefix : str, default=None
            The path prefix to save result as ``path_prefix+'_'+region.label+'_'+ str(state_u.value)+'_'+str(state_v.value)+'.tif'.
            If None, the result is returned.
            Note that if ``path_prefix is not None``, ``lul`` must be LandUseLayer

        Returns
        -------
        tp : dict(Region:[J,P_v__u_Y])
            Dict of regional results with corresponding region as key. Th regional results are ``J``, a ndarray of shape (n_samples,)
            which is the element indices in the flattened map and ``P_v__u_Y``, the transition probabilities whose columns correspond
            to transition matrix argument : ``palette_v``.
        """

        if self.verbose > 0:
            print('\n## Territory TPE\n')

        if masks is None:
            masks = {region: None for region in self.regions}

        distances_to_states = {}

        tp = {}

        for region in self.regions:

            if path_prefix is not None:
                path_prefix += '_' + str(region.label)

            tp[region] = region.transition_probabilities(transition_matrix=transition_matrices[region],
                                                         lul=lul,
                                                         mask=masks[region],
                                                         distances_to_states=distances_to_states,
                                                         path_prefix=path_prefix)

        if self.verbose > 0:
            print('Territory transition probabilities estimation done.\n')

        if path_prefix is None:
            return (tp)

    def allocate(self,
                 transition_matrices,
                 lul,
                 masks=None,
                 path=None):
        """
        Allocate.

        Parameters
        ----------
        transition_matrices : dict(Region:TransitionMatrix)
            Dict of transition matrix with the corresponding region as key.

        lul : LandUseLayer or ndarray
            The studied land use layer.

        masks : dict(Region:MaskLayer), default=None
            Dict of masks layer with the corresponding region as key. If None, the whole map is used for each region.

        path : str, default=None
            The path to save result as a tif file.
            If None, the allocation is only saved within `lul`, if `lul` is a ndarray.
            Note that if ``path`` is not ``None``, ``lul`` must be LandUseLayer.

        Returns
        -------
        lul_allocated : LandUseLayer
            Only returned if ``path`` is not ``None``. The allocated map as a land use layer.
        """

        if self.verbose > 0:
            print('\n## Territory allocate\n')

        if masks is None:
            masks = {region: None for region in self.regions}

        distances_to_states = {}

        lul_data = lul.get_data().copy()

        for region in self.regions:
            region.allocate(transition_matrix=transition_matrices[region],
                            lul=lul_data,
                            lul_origin=lul,
                            mask=masks[region],
                            distances_to_states=distances_to_states,
                            path=None)
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

    def multisteps_allocation(self,
                              n,
                              transition_matrices,
                              lul,
                              masks=None,
                              path_prefix=None):
        """
        Multisteps allocation

        Parameters
        ----------
        n : int
            Number of steps.

        transition_matrices : dict(Region:TransitionMatrix)
            Dict of transition matrix with the corresponding region as key.

        lul : LandUseLayer or ndarray
            The studied land use layer.

        masks : dict(Region:MaskLayer), default=None
            Dict of masks layer with the corresponding region as key. If None, the whole map is used for each region.

        path_prefix : str, default=None
            The path to save every allocated step map as ``path_prefix+'_'+str(i)+'.tif'``.
            If None, the allocation is only saved within `lul`, if `lul` is a ndarray.
            Note that if ``path`` is not ``None``, ``lul`` must be LandUseLayer.

        Returns
        -------
        lul_allocated : LandUseLayer
            The allocated map as a land use layer.
        """



        multisteps_transition_matrices = {region: tm.multisteps(n) for region, tm in transition_matrices.items()}

        lul_step = lul

        for i in range(n):

            if self.verbose > 0:
                print('\n# Territory multisteps allocate '+str(i)+'\n')

            if isinstance(path_prefix, str):
                path_step = path_prefix + '_' + str(i) + '.tif'
            elif callable(path_prefix):
                path_step = path_prefix(i)

            lul_step = self.allocation(transition_matrices=multisteps_transition_matrices,
                                       lul=lul_step,
                                       masks=masks,
                                       path=path_step)

            if self.verbose > 0:
                print('Territory multisteps allocate '+str(i)+' done.\n')

        if path_prefix is not None:
            return lul_step
