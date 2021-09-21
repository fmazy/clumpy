#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:03:33 2021

@author: frem
"""

from ..density_estimation import GKDE

import numpy as np
from ..density_estimation._density_estimator import DensityEstimator
from ..density_estimation import _methods

# READ ME
# Transition Probability Estimators (TPE) must have these methods :
#   - fit()
#   - transition_probability()
#   - _check()

class TransitionProbabilityEstimator():
    """
    Transition probability estimator base class.
    """
    def __init__(self,
                 verbose=0,
                 verbose_heading_level=1):
        self.verbose = verbose
        self.verbose_heading_level = verbose_heading_level

    def check(self):
        """
        Check the density estimators uniqueness.
        """
        self._check()
