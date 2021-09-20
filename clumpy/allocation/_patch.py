#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class Patch():
    """
    Patch parameters object. Should prefer the child BootStrapPatch.

    Parameters
    ----------
    method : str
        Patch caracteristics draws method.
    """
    def __init__(self,
                 neighbors_structure, 
                 avoid_aggregation = True,
                 nb_of_neighbors_to_fill = 3,
                 proceed_even_if_no_probability = True,
                 n_tries_target_sample = 'rook'):
        self.neighbors_structure = neighbors_structure
        self.avoid_aggregation = avoid_aggregation
        self.nb_of_neighbors_to_fill = nb_of_neighbors_to_fill
        self.proceed_even_if_no_probability = proceed_even_if_no_probability
        self.n_tries_target_sample = n_tries_target_sample
    
    def sample(self, n):
        """
        draws patches.
        """
        return(self._sample(n))
    
    def target_sample(self, n):
        """
        Draw areas and eccentricities according to a targeted total area.
    
        Parameters
        ----------
        n
    
        Returns
        -------
        None.
    
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
    areas : array-like of shape (n_patches,)
        Array of areas.
    eccentricities : array-like of shape (n_patches,)
        Array of eccentricities which correspond to areas.
    """
    def __init__(self,
                 areas, 
                 eccentricities,
                 neighbors_structure = 'rook',
                 avoid_aggregation = True,
                 nb_of_neighbors_to_fill = 3,
                 proceed_even_if_no_probability = True,
                 n_tries_target_sample = 1000):
        
        super().__init__(neighbors_structure = neighbors_structure,
                         avoid_aggregation = avoid_aggregation,
                         nb_of_neighbors_to_fill = nb_of_neighbors_to_fill,
                         proceed_even_if_no_probability = proceed_even_if_no_probability,
                         n_tries_target_sample=n_tries_target_sample)
        
        self.areas = areas
        self.eccentricities = eccentricities
        
        self.area_mean = np.mean(areas)
        self.eccentricities_mean = np.mean(eccentricities)
    
    def _sample(self, n):
        idx = np.random.choice(self.areas.size, n, replace=True)
        
        return(self.areas[idx], self.eccentricities[idx])
    
    

# class TransitionPatches():
#     """
#     Patches informations for a given transition

#     Parameters
#     ----------
#     patches_isl : BootstrapPatchParams, default=None
#         The island patch parameters. Unexpected to be ``None`` if ``ratio>0``.
#     patches_exp : BootstrapPatchParams, default=None
#         The expansion patch parameters. Unexpected to be ``None`` if ``ratio<1``.
#     ratio_isl : float
#         The ratio of island areas over the whole areas to allocate.
#     """
#     def __init__(self,
#                  patches_isl = None,
#                  patches_exp = None,
#                  ratio_isl = 1.0):
        
#         if ratio_isl < 1.0 and patches_exp is None:
#             raise(ValueError('Unexpected patches_exp for such a ratio_isl.'))
#         if ratio_isl > 0.0 and patches_isl is None:
#             raise(ValueError('Unexpected patches_isl for such a ratio_isl.'))
        
#         self.patches_isl = patches_isl
#         self.patches_exp = patches_exp
#         self.ratio_isl = ratio_isl
        
