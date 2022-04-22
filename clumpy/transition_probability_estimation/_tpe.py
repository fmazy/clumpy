#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:03:33 2021

@author: frem
"""

import numpy as np

from .._base import Palette
from ..tools._console import title_heading

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
                 initial_state,
                 verbose=0,
                 verbose_heading_level=1):
        self.initial_state = initial_state
        self.verbose = verbose
        self.verbose_heading_level = verbose_heading_level



    



