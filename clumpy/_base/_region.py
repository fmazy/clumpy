#!/usr/bin/env python3
# -*- coding: utf-8 -*

import numpy as np
from copy import deepcopy

from ..layer import LandUseLayer, ProbaLayer
from ._transition_matrix import TransitionMatrix, load_transition_matrix
from ..tools._path import path_split
from ..tools._console import title_heading
from . import Land

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
                 value,
                 verbose=0,
                 verbose_heading_level=1):
        self.label = label
        self.value = value
        self.lands = []
        self.verbose = verbose
        self.verbose_heading_level = verbose_heading_level
        
    def __repr__(self):
        return 'Region('+self.label+')'
            
    def get_lands_states(self):
        
        return [l.state for l in self.lands]
    
    def add_land(self, land):
        if land not in self.lands:
            if land.state not in self.get_lands_states():
                self.lands.append(land)
            else:
                Warning('The land value is already in.')
        else:
            Warning('The land is already in.')
        
        return(self)
    
    def add_lands(self, lands):
        self.lands = lands
        return self
    
    def get_land_by_state(self, state):
        states = self.get_lands_states()
        return(self.lands[states.index(state)])
    
    def get_land(self, info):
        if type(info) is int:
            return(self.get_land_by_state(info))
    
    # def check(self, objects=None):
    #     """
    #     Check the unicity of objects.
    #     Notably, estimators uniqueness are checked to avoid malfunctioning during transition probabilities estimation.
    #     """
    #     if objects is None:
    #         objects = []
        
        
    #     for land in self.values():
    #         if land in objects:
    #             raise(ValueError("Land objects must be different."))
    #         else:
    #             objects.append(land)
                
    #         land.check(objects=objects)
        

    # def fit(self):
    #     """
    #     Fit the region.

    #     Parameters
    #     ----------
    #     lul_initial : LandUseLayer
    #         The initial land use.

    #     lul_final : LandUseLayer
    #         The final land use.

    #     mask : MaskLayer, default = None
    #         The region mask layer. If ``None``, the whole area is studied.

    #     distances_to_states : dict(State:ndarray), default={}
    #         The distances matrix to key state. Used to improve performance.

    #     Returns
    #     -------
    #     self
    #     """
    #     if self.verbose > 0:
    #         print(title_heading(self.verbose_heading_level) + 'Region ' + self.label + ' fitting\n')

    #     for state, land in self.items():
    #         land.fit()


        # return (self)

    # def compute_transition_matrix(self,
    #                               lul_initial='initial',
    #                               lul_final='final',
    #                               mask='calibration'):
    #     """
    #     Compute transition matrix

    #     Parameters
    #     ----------
    #     lul_initial : LandUseLayer
    #         The initial land use.

    #     lul_final : LandUseLayer
    #         The final land use.

    #     mask : MaskLayer, default = None
    #         The region mask layer. If ``None``, the whole area is studied.

    #     Returns
    #     -------
    #     tm : TransitionMatrix
    #         The computed transition matrix.
    #     """
        
    #     if isinstance(lul_initial, str):
    #         lul_initial = self.get_lul(lul_initial)
        
    #     if isinstance(lul_final, str):
    #         lul_final = self.get_lul(lul_final)
            
    #     if isinstance(mask, str):
    #         mask = self.get_mask(mask)
        
    #     tm = None
        
    #     for state, land in self.items():
    #         tm_to_merge = land.compute_transition_matrix(lul_initial=lul_initial,
    #                                                      lul_final=lul_final,
    #                                                      mask=mask)
    #         if tm is None:
    #             tm = tm_to_merge
    #         else:
    #             tm.merge(tm=tm_to_merge, inplace=True)

    #     return (tm)

    # def transition_probabilities(self,
    #                              lul='start',
    #                              mask=None,
    #                              effective_transitions_only=True,
    #                              territory_format=False):
    #     """
    #     Compute transition probabilities.

    #     Parameters
    #     ----------
    #     transition_matrix : TransitionMatrix
    #         The requested transition matrix.

    #     lul : LandUseLayer
    #         The studied land use layer.

    #     mask : MaskLayer, default = None
    #         The region mask layer. If ``None``, the whole map is studied.

    #     distances_to_states : dict(State:ndarray), default={}
    #         The distances matrix to key state. Used to improve performance.

    #     path_prefix : str, default=None
    #         The path prefix to save result as ``path_prefix+'_'+ str(state_u.value)+'_'+str(state_v.value)+'.tif'.
    #         If None, the result is returned.

    #     Returns
    #     -------
    #     J : dict(State:ndarray of shape (n_samples,))
    #         Only returned if ``path_prefix=False``. Element indexes in the flattened
    #         matrix for each state.

    #     P_v__u_Y : dict(State:ndarray of shape (n_samples, len(palette_v)))
    #         Only returned if ``path_prefix=False``. The transition probabilities of each elements for each state. Ndarray columns are
    #         ordered as ``palette_v``.
    #     """

    #     if isinstance(lul, str):
    #         lul = self.get_lul(lul)
        
    #     if mask is None:
    #         mask = self.get_mask('allocation')
        
    #     p = {}
        
    #     for state, land in self.items():
    #         p[int(state)] = land.transition_probabilities(
    #             lul=lul,
    #             mask=mask,
    #             effective_transitions_only=effective_transitions_only,
    #             territory_format=False)
        
    #     if not territory_format:
    #         return(p)
    #     else:
    #         return({self.label : p})
    
    # def transition_probabilities_layer(self,
    #                                    effective_transitions_only=True):
        
    #     if self.verbose > 0:
    #         print(title_heading(self.verbose_heading_level) + 'Region ' + self.label + ' TPE\n')
        
    #     lul = self.get_lul('start')
        
    #     proba_layer = ProbaLayer(np.ndarray(shape=(0,) + lul.shape),
    #                              label="",
    #                              final_states=[],
    #                              geo_metadata = deepcopy(lul.geo_metadata))
                
    #     for state, land in self.items():
    #         proba_layer__land = land.transition_probabilities_layer(
    #             effective_transitions_only=effective_transitions_only)
    #         proba_layer = proba_layer.fusion(proba_layer__land)
                
    #     return(proba_layer)
        
    # def allocate(self,
    #              lul='start',
    #              lul_origin=None):
        
    #     if self.verbose > 0:
    #         print(title_heading(self.verbose_heading_level) + 'Region ' + self.label + ' Allocation\n')
        
    #     if isinstance(lul, str):
    #         lul = self.get_lul(lul).copy()
            
    #     if lul_origin is None:
    #         lul_origin = lul.copy()
        
    #     proba_layer = ProbaLayer(np.ndarray(shape=(0,) + lul.shape),
    #                              label="",
    #                              final_states=[],
    #                              geo_metadata = deepcopy(lul.geo_metadata))
        
    #     for land in self.values():
    #         lul, proba_layer__land = land.allocate(lul=lul,
    #                                     lul_origin=lul_origin)
    #         proba_layer = proba_layer.fusion(proba_layer__land)
        
    #     return(lul, proba_layer)
            
    
    # def allocate(self,
    #              p,
    #              lul,
    #              lul_origin=None,
    #              distances_to_states={}):
    #     """
    #     allocation.

    #     Parameters
    #     ----------
    #     transition_matrix : TransitionMatrix
    #         The requested transition matrix.

    #     lul : LandUseLayer or ndarray
    #         The studied land use layer. If ndarray, the matrix is directly edited (inplace).

    #     lul_origin : LandUseLayer
    #         Original land use layer. Usefull in case of regional allocations. If ``None``, the  ``lul`` layer is copied.

    #     mask : MaskLayer, default = None
    #         The region mask layer. If ``None``, the whole map is studied.

    #     distances_to_states : dict(State:ndarray), default={}
    #         The distances matrix to key state. Used to improve performance.

    #     path : str, default=None
    #         The path to save result as a tif file.
    #         If None, the allocation is only saved within `lul`, if `lul` is a ndarray.
    #         Note that if ``path`` is not ``None``, ``lul`` must be LandUseLayer.

    #     path_prefix_transition_probabilities : str, default=None
    #         The path prefix to save transition probabilities.

    #     Returns
    #     -------
    #     lul_allocated : LandUseLayer
    #         Only returned if ``path`` is not ``None``. The allocated map as a land use layer.
    #     """

    #     if lul is None:
    #         lul = 'start'
        
    #     if isinstance(lul, str):
    #         lul = self.get_lul(lul)
        
    #     if isinstance(lul, LandUseLayer):
    #         lul_data = lul.get_data()
    #     else:
    #         lul_data = lul
        
    #     if lul_origin is None:
    #         lul_origin_data = lul_data.copy()
    #     else:
    #         lul_origin_data = lul_origin
            
    #     for initial_state, [J, P_v__u_Y, final_states] in p.items():
    #         self[initial_state].allocate(J=J,
    #                                      P_v__u_Y=P_v__u_Y,
    #                                      final_states=final_states,
    #                                      lul=lul_data,
    #                                      lul_origin=lul_origin_data,
    #                                      distances_to_states=distances_to_states)
        
    #     return(lul_data)
    
    # def allocate_layer(self,
    #                    path,
    #                    p,
    #                    lul,
    #                    lul_origin=None,
    #                    distances_to_states={}):
    #     lul_data = self.allocate(p=p,
    #                              lul=lul,
    #                              lul_origin=lul_origin,
    #                              distances_to_states=distances_to_states)
        
    #     alloc_layer = LandUseLayer(path=path,
    #                                data=lul_data,
    #                                copy_geo=lul,
    #                                palette=lul.palette)
    #     return(alloc_layer)
        
    # def add_land(self, land):
    #     """
    #     Add a land for a given state.

    #     Parameters
    #     ----------
    #     state : State
    #         The initial state.
        
    #     land : Land
    #         The Land object.

    #     Returns
    #     -------
    #     self
    #     """
    #     land.region = self
        
    #     self[land.state] = land
        

    #     return (self)
    
    # def make(self, palette, **params):
                
    #     self = {}
        
        
    #     if 'transition_matrix' in params.keys():
    #         transition_matrix = load_transition_matrix(path=params['transition_matrix'],
    #                                                    palette=palette)
                
    #     for state_u in transition_matrix.palette_u:
    #         land = Land(state=state_u,
    #                     verbose=self.verbose,
    #                     verbose_heading_level=4)
            
    #         land_params = params.copy()
    #         # land_params is appended with specific parameters of the land
    #         if 'lands' in land_params.keys():
    #             if str(state_u.value) in land_params['states'].keys():
    #                 for key in land_params['states'][str(state_u.value)]:
    #                     land_params[key] = land_params['states'][str(state_u.value)][key]
                                
    #         land.make(palette, **land_params)
            
    #         self.add_land(land=land)
