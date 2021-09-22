#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class Patch():
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
                 neighbors_structure = 'rook',
                 avoid_aggregation = True,
                 nb_of_neighbors_to_fill = 3,
                 proceed_even_if_no_probability = True,
                 n_tries_target_sample = 10**3,
                 equi_neighbors_proba = False):
        self.neighbors_structure = neighbors_structure
        self.avoid_aggregation = avoid_aggregation
        self.nb_of_neighbors_to_fill = nb_of_neighbors_to_fill
        self.proceed_even_if_no_probability = proceed_even_if_no_probability
        self.n_tries_target_sample = n_tries_target_sample
        self.equi_neighbors_proba = equi_neighbors_proba

        # for compatibility, set mean area and eccentricities to 1.0 by default.
        self.area_mean = 1.0
        self.eccentricities_mean = 1.0

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
        
class BootstrapPatch(Patch):
    """
    Bootstrap patch parameters object.

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
                 neighbors_structure = 'rook',
                 avoid_aggregation = True,
                 nb_of_neighbors_to_fill = 3,
                 proceed_even_if_no_probability = True,
                 n_tries_target_sample = 1000,
                 equi_neighbors_proba=False):
        
        super().__init__(neighbors_structure = neighbors_structure,
                         avoid_aggregation = avoid_aggregation,
                         nb_of_neighbors_to_fill = nb_of_neighbors_to_fill,
                         proceed_even_if_no_probability = proceed_even_if_no_probability,
                         n_tries_target_sample=n_tries_target_sample,
                         equi_neighbors_proba=equi_neighbors_proba)

    def _sample(self, n):
        idx = np.random.choice(self.areas.size, n, replace=True)
        
        return(self.areas[idx], self.eccentricities[idx])

    def set(self,
            areas,
            eccentricities):
        """
        Set areas and eccentricities.

        Parameters
        ----------
        areas : array-like of shape (n_patches,)
            Array of areas.
        eccentricities : array-like of shape (n_patches,)
            Array of eccentricities which correspond to areas.

        Returns
        -------
        self
        """

        if areas.size > 0 and areas.size == eccentricities.size:
            self.areas = areas
            self.eccentricities = eccentricities

            self.area_mean = np.mean(areas)
            self.eccentricities_mean = np.mean(eccentricities)

        return(self)

    def crop_areas(self,
                   min_area=-np.inf,
                   max_area=np.inf,
                   inplace=True):
        """
        Crop areas.

        Parameters
        ----------
        min_area : float, default=-np.inf
            Minimum area threshold.
        max_area : float, default=np.inf
            Maximum area threshold.

        Returns
        -------
        self
        """
        idx = self.areas >= min_area & self.areas <= max_area

        if inplace:
            self.areas = self.areas[idx]
            self.eccentricities = self.eccentricities[idx]
        else:
            return(BootstrapPatch().set(areas=self.areas[idx],
                                        eccentricities=self.eccentricities[idx]))
