# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import lognorm as scipy_lognorm

from ._patcher import Patcher

class LogNormPatcher(Patcher):
    def __init__(self,
                 initial_state,
                 final_state,
                 area_mean=10.0,
                 area_var=5.0,
                 eccentricity=0.5,
                 neighbors_structure='rook',
                 avoid_aggregation=True,
                 nb_of_neighbors_to_fill=3,
                 proceed_even_if_no_probability=True,
                 n_tries_target_sample=1000,
                 equi_neighbors_proba=False):
        super().__init__(initial_state = initial_state,
                         final_state = final_state,
                         neighbors_structure = neighbors_structure,
                         avoid_aggregation = avoid_aggregation,
                         nb_of_neighbors_to_fill = nb_of_neighbors_to_fill,
                         proceed_even_if_no_probability = proceed_even_if_no_probability,
                         n_tries_target_sample=n_tries_target_sample,
                         equi_neighbors_proba=equi_neighbors_proba)

        self.area_mean = area_mean
        self.area_var = area_var
        self.eccentricity = eccentricity

    def _sample(self, n):
        eccentricities = np.ones(n) * self.eccentricity
        
        # mu = np.log2(self.area_mean) - 0.5 * np.log2(1 + self.area_var / self.area_mean**2)
        # delta = np.sqrt(np.log2(1 + self.area_var/self.area_mean**2))
        
        E = self.area_mean
        V = self.area_var
        mu = np.log(E**2 / np.sqrt(V+E**2))
        delta = np.sqrt(np.log(V/E**2 + 1))
        
        areas = np.random.lognormal(mean=mu, sigma=delta, size=n)
        
        # areas = scipy_lognorm.rvs(s=delta,
                                  # scale=np.exp(mu),
                                  # loc=mu,
                                  # size=n)
        
        # areas = scipy_lognorm.rvs(s=n, 
                                  # loc=self.area_mean,
                                  # scale=self.area_var)
        
        if n == 1:
            areas = np.array([areas])
            
        areas[areas < 1] = 1
        
        return (areas, eccentricities)