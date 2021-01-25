#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 18:29:51 2021

@author: frem
"""
import numpy as np

def weighted_kolmogorov_smirnov(edf, cdf, alpha=1):
    return(np.max(np.abs(edf-cdf)*np.power(edf*(1-edf),alpha)))