#!/usr/bin/env python3
# -*- coding: utf-8 -*

import numpy as np

from ._layer import LandUseLayer
from ._transition_matrix import TransitionMatrix, load_transition_matrix
from ..tools._path import path_split
from ..tools._console import title_heading
from . import Land
from ._feature import Features

class Region():
    """
    Define a region.

    Parameters
    ----------
    label : str
        The region's label. It should be unique.

    verbose : int, default=0
        Verbosity level.

    verbose_heading_level : int, default=1
        Verbose heading level for markdown titles. If ``0``, no markdown title are printed.
    """

    def __init__(self,
                 label,
                 verbose=0,
                 verbose_heading_level=1):
        self.label = label
        self.verbose = verbose
        self.verbose_heading_level = verbose_heading_level

        self.lands = {}
        
        self.territory = None
        # self.features = None
        self.calibrator = None
        self.lul = {}
        self.mask = {}
        self.transition_matrix = None

    def __repr__(self):
        return (self.label)

    def add_land(self, land):
        """
        Add a land for a given state.

        Parameters
        ----------
        state : State
            The initial state.
        
        land : Land
            The Land object.

        Returns
        -------
        self
        """
        land.region = self
        
        self.lands[land.state] = land
        

        return (self)
    
    def make(self, palette, **params):
                
        self.lands = {}
        
        
        if 'transition_matrix' in params.keys():
            transition_matrix = load_transition_matrix(path=params['transition_matrix'],
                                                       palette=palette)
                
        for state_u in transition_matrix.palette_u:
            land = Land(state=state_u,
                        verbose=self.verbose,
                        verbose_heading_level=4)
            
            land_params = params.copy()
            # land_params is appended with specific parameters of the land
            if 'lands' in land_params.keys():
                if str(state_u.value) in land_params['states'].keys():
                    for key in land_params['states'][str(state_u.value)]:
                        land_params[key] = land_params['states'][str(state_u.value)][key]
                                
            land.make(palette, **land_params)
            
            self.add_land(land=land)
        

    def _check_density_estimators(self, density_estimators=[]):
        """
        Check the density estimators uniqueness.
        """
        for state, land in self.lands.items():
            density_estimators = land._check_density_estimators(density_estimators=density_estimators)

        return (density_estimators)

    def _check_feature_selectors(self, feature_selectors=[]):
        """
        check the feature selectors uniqueness.
        """
        for state, land in self.lands.items():
            feature_selectors = land._check_feature_selectors(feature_selectors=feature_selectors)

        return (feature_selectors)
    
    # def set_features(self, features):
    #     self.features = features
    
    # def get_features(self):
    #     if self.features is None:
    #         return(self.territory.get_features())
    #     else:
    #         return(self.features)
    
    def set_calibrator(self, calibrator):
        self.calibrator = calibrator
    
    def get_calibrator(self):
        if self.calibrator is None:
            return(self.territory.get_calibrator())
        else:
            return(self.calibrator)
    
    def set_lul(self, lul, kind):
        self.lul[kind] = lul
    
    def get_lul(self, kind):
        if kind not in self.lul.keys():
            return(self.territory.get_lul(kind))
        else:
            return(self.lul[kind])
    
    def set_mask(self, mask, kind):
        self.mask[kind] = mask
    
    def get_mask(self, kind):
        if kind not in self.mask.keys():
            return(self.territory.get_mask(kind))
        else:
            return(self.mask[kind])
    
    def set_transition_matrix(self, tm):
        self.transition_matrix = tm
    
    def get_transition_matrix(self):
        if self.transition_matrix is None:
            return(self.territory.get_transition_matrix())
        else:
            return(self.transition_matrix)
    
    def check(self):
        """
        Check the Region object through lands checks.
        Notably, estimators uniqueness are checked to avoid malfunctioning during transition probabilities estimation.
        """
        self._check_density_estimators()
        self._check_feature_selectors()
        
        for state, land in self.lands.items():
            land.state = state

    def fit(self,
            lul_initial,
            lul_final,
            mask=None,
            distances_to_states={}):
        """
        Fit the region.

        Parameters
        ----------
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
            print(title_heading(self.verbose_heading_level) + 'Region ' + self.label + ' fitting\n')

        for state, land in self.lands.items():
            land.fit(lul_initial=lul_initial,
                     lul_final=lul_final,
                     mask=mask,
                     distances_to_states=distances_to_states)

        if self.verbose > 0:
            print('Region ' + self.label + ' fitting done.\n')

        return (self)

    def transition_matrix(self,
                          lul_initial,
                          lul_final,
                          mask=None):
        """
        Compute transition matrix

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
        tm : TransitionMatrix
            The computed transition matrix.
        """
        tm = None
        for state, land in self.lands.items():
            tm_to_merge = land.transition_matrix(lul_initial=lul_initial,
                                                 lul_final=lul_final,
                                                 mask=mask)
            if tm is None:
                tm = tm_to_merge
            else:
                tm.merge(tm=tm_to_merge, inplace=True)

        return (tm)

    def transition_probabilities(self,
                                 transition_matrix,
                                 lul,
                                 mask=None,
                                 distances_to_states={},
                                 path_prefix=None,
                                 copy_geo=None):
        """
        Compute transition probabilities.

        Parameters
        ----------
        transition_matrix : TransitionMatrix
            The requested transition matrix.

        lul : LandUseLayer
            The studied land use layer.

        mask : MaskLayer, default = None
            The region mask layer. If ``None``, the whole map is studied.

        distances_to_states : dict(State:ndarray), default={}
            The distances matrix to key state. Used to improve performance.

        path_prefix : str, default=None
            The path prefix to save result as ``path_prefix+'_'+ str(state_u.value)+'_'+str(state_v.value)+'.tif'.
            If None, the result is returned.

        Returns
        -------
        J : dict(State:ndarray of shape (n_samples,))
            Only returned if ``path_prefix=False``. Element indexes in the flattened
            matrix for each state.

        P_v__u_Y : dict(State:ndarray of shape (n_samples, len(palette_v)))
            Only returned if ``path_prefix=False``. The transition probabilities of each elements for each state. Ndarray columns are
            ordered as ``palette_v``.
        """

        J = {}
        P_v__u_Y = {}

        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level) + 'Region ' + str(self.label) + ' TPE\n')

        if isinstance(lul, LandUseLayer):
            copy_geo = lul

        for state, land in self.lands.items():

            if self.verbose > 0:
                print('state ' + str(state))

            if path_prefix is not None:
                land_path_prefix = path_prefix + '_' + str(state.value)
            else:
                land_path_prefix = None

            ltp = land.transition_probabilities(transition_matrix=transition_matrix.extract(infos=[state]),
                                                lul=lul,
                                                mask=mask,
                                                distances_to_states=distances_to_states,
                                                path_prefix=land_path_prefix,
                                                copy_geo=copy_geo)

            if path_prefix is None:
                J[state] = ltp[0]
                P_v__u_Y[state] = ltp[1]

        if self.verbose > 0:
            print('Region ' + str(self.label) + ' TPE done.\n')

        return (J, P_v__u_Y)

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
            The requested transition matrix.

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

        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level) + 'Region ' + str(self.label) + ' allocate\n')

        if lul_origin is None:
            lul_origin = lul

        if isinstance(lul_origin, LandUseLayer):
            lul_origin_data = lul_origin.get_data()
            copy_geo = lul_origin
        else:
            lul_origin_data = lul_origin

        if isinstance(lul, LandUseLayer):
            lul_data = lul.get_data().copy()
        else:
            lul_data = lul

        for state, land in self.lands.items():

            if path_prefix_transition_probabilities is not None:
                land_path_prefix_transition_probabilities = path_prefix_transition_probabilities + '_' + str(state.value)
            else:
                land_path_prefix_transition_probabilities = None

            land.allocate(transition_matrix=transition_matrix.extract(infos=[state]),
                          lul=lul_data,
                          lul_origin=lul_origin_data,
                          mask=mask,
                          distances_to_states=distances_to_states,
                          path=None,
                          path_prefix_transition_probabilities=land_path_prefix_transition_probabilities,
                          copy_geo=copy_geo)
            # Note that the path is set to None in the line above in order to allocate through all regions and save in a second time !

        if self.verbose > 0:
            print('Region ' + str(self.label) + ' allocate done.\n')

        if path is not None:
            folder_path, file_name, file_ext = path_split(path)
            return (LandUseLayer(label=file_name,
                                 data=lul_data,
                                 copy_geo=copy_geo,
                                 path=path,
                                 palette=lul_origin.palette))
