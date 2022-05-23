# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm as scipy_norm

from ._patcher import Patcher

class GaussianPatcher(Patcher):
    def __init__(self,
                 initial_state,
                 final_state,
                 area_mean=10.0,
                 area_cov=5.0,
                 eccentricity=0.5,
                 neighbors_structure='rook',
                 avoid_aggregation=True,
                 nb_of_missing_to_fill=1,
                 proceed_even_if_no_probability=True,
                 n_tries_target_sample=1000,
                 equi_neighbors_proba=False):
        super().__init__(initial_state = initial_state,
                         final_state = final_state,
                         neighbors_structure = neighbors_structure,
                         avoid_aggregation = avoid_aggregation,
                         nb_of_missing_to_fill = nb_of_missing_to_fill,
                         proceed_even_if_no_probability = proceed_even_if_no_probability,
                         n_tries_target_sample=n_tries_target_sample,
                         equi_neighbors_proba=equi_neighbors_proba)

        self.area_mean = area_mean
        self.area_cov = area_cov
        self.eccentricity = eccentricity

    def _sample(self, n):
        eccentricities = np.ones(n) * self.eccentricity
        areas = scipy_norm.rvs(loc=self.area_mean,
                               scale=self.area_cov,
                               size=n)
        areas[areas < 1] = 1

        return (areas, eccentricities)