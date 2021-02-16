#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:33:19 2021

@author: frem
"""

import numpy as np
cimport numpy as np

cpdef double mcdf(int n):
    cdef double a = 0.0
    cdef double b = 1.0
    cdef int i
    
    for i in range(n):
        a, b = a + b, a
    return(a)
