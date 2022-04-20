#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


from copy import deepcopy

from ..tools._data import np_drop_duplicates_from_column

structures = {
    'queen' : np.ones((3, 3)),
    'rook' : np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])
}

class Patchers(dict):
    
    # def __init__(self, state):
    #     self.state = self.state
    
    def add_patcher(self, patcher):
        self[patcher.state] = patcher
    
    def check(self, objects=None):
        if objects is None:
            objects = []
            
        for state, patcher in self.items():
            if patcher in objects:
                raise(ValueError("Patchers objects must be different."))
            else:
                objects.append(patcher)
            
            if int(state) != int(patcher.final_state):
                raise(ValueError("Patchers keys does not correspond to Patcher().initial_state values."))
            
    
    def fit(self,
            J,
            V,
            shape):
        
        for state, patch in self.items():
            patch.fit(J,
                      V,
                      shape)

class Patcher():
    """
    Patch parameters object. Useful for developers.

    Parameters
    ----------
    neighbors_structure : {'rook', 'queen'}, default='rook'
        The neighbors structure.

    avoid_aggregation : bool, default=True
        If ``True``, the patcher will avoid patch aggregations to respect expected patch areas.

    nb_of_neighbors_to_fill : int, default=3
        The patcher will allocate cells whose the number of allocated neighbors is greater than this integer
        (according to the specified ``neighbors_structure``)

    proceed_even_if_no_probability : bool, default=True
        The patcher will allocate even if the neighbors have no probabilities to transit.

    n_tries_target_sample : int, default=10**3
        Number of tries to draw samples in a biased way in order to approach the mean area.

    equi_neighbors_proba : bool, default=False
        If ``True``, all neighbors have the equiprobability to transit.
    """
    def __init__(self,
                 initial_state,
                 final_state,
                 neighbors_structure = 'rook',
                 avoid_aggregation = True,
                 nb_of_neighbors_to_fill = 3,
                 proceed_even_if_no_probability = True,
                 n_tries_target_sample = 10**3,
                 equi_neighbors_proba = False):
        self.initial_state = initial_state
        self.final_state = final_state
        self.neighbors_structure = neighbors_structure
        self.avoid_aggregation = avoid_aggregation
        self.nb_of_neighbors_to_fill = nb_of_neighbors_to_fill
        self.proceed_even_if_no_probability = proceed_even_if_no_probability
        self.n_tries_target_sample = n_tries_target_sample
        self.equi_neighbors_proba = equi_neighbors_proba

        # for compatibility, set mean area and eccentricities to 1.0 by default.
        self.area_mean = 1.0
        self.eccentricities_mean = 1.0
        
    
    def __repr__(self):
        return("Patcher("+str(self.initial_state)+"->"+str(self.final_state)+")")
    
    def copy(self):
        return(deepcopy(self))
    
    def sample(self, n):
        """
        draws patches.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        areas : ndarray of shape (n_samples,)
            The samples areas.
        eccentricities : ndarray of shape (n_samples,)
            The samples eccentricities.
        """
        return(self._sample(n))
    
    def target_sample(self, n):
        """
        Draw areas and eccentricities according to a targeted total area (biased sample).
    
        Parameters
        ----------
        n : int
            The number of samples.
    
        Returns
        -------
        areas : ndarray of shape (n_samples,)
            The samples areas.
        eccentricities : ndarray of shape (n_samples,)
            The samples eccentricities.
        """
        n_try = 0
            
        best_areas = None
        best_eccentricities = None
        best_relative_error = np.inf
        
        total_area_target = self.area_mean * n
        
        while n_try < self.n_tries_target_sample:
            n_try += 1
            
            areas, eccentricities = self.sample(n)
            
            relative_error = np.abs(total_area_target - areas.sum()) / total_area_target
            
            if relative_error < best_relative_error:
                best_relative_error = relative_error
                best_areas = areas
                best_eccentricities = eccentricities
        
        return(best_areas, best_eccentricities)

    def fit(self,
            J,
            V,
            shape):
        return(self)


