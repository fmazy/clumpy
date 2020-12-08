#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:39:35 2020

@author: frem
"""
import numpy as np

def adjust_probabilities(P, f):
    
    if np.abs(f.sum() - 1) > 10**-10:
        raise(TypeError("Uncorrect scenario. The sum of frequencies should be equal to one."))
    
    
    P_mean = P.mean(axis=0)
    
    num = (f / P_mean)[None,:] * P
    den = ((f / P_mean)[None,:] * P).sum(axis=1)
    
    return(num/den[:,None])
        