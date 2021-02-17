#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:33:19 2021

@author: frem
"""

import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

# cpdef np.ndarray[double, ndim=2] pdf_to_mcdf(pdf,
#                 np.ndarray[double, ndim=1] X,
#                 int column,
#                 np.ndarray[double, ndim=1] a,
#                 np.ndarray[double, ndim=1] b,
#                 long n_mc=1000,
#                 args={}):
    
#     # the number of elements
#     cdef np.npy_intp n_samples = X.shape[0]
    
#     # the number of features
#     cdef np.npy_intp n_features = a.size
    
#     cdef np.ndarray[long, ndim=1] b_columns = np.arange(n_features)
#     b_columns = np.delete(b_columns, column)
    
#     # create the X prime
#     cdef np.ndarray[double, ndim=2] X_prime = np.zeros((n_samples, n_features), dtype=float)
    
#     X_prime[:,column] = X
#     X_prime[:,b_columns] = b[b_columns]
    
#     cdef np.ndarray[double, ndim=1] mcdf = pdf_to_cdf(pdf,
#                                                       X_prime,
#                                                       a,
#                                                       b,
#                                                       n_mc,
#                                                       args)
    
#     return(mcdf)

# cpdef np.ndarray[double, ndim=1] pdf_to_cdf(pdf,
#                np.ndarray[double, ndim=2] X,
#                np.ndarray[double, ndim=1] a,
#                np.ndarray[double, ndim=1] b,
#                long n_mc=1000,
#                args={}):
    
#     # the number of elements
#     cdef np.npy_intp n_samples = X.shape[0]
    
#     # the number of features
#     cdef np.npy_intp n_features = X.shape[1]
    
#     # generate the monte carlo elements
#     cdef np.ndarray[double, ndim=2] X_mc = _generate(n_mc, n_features) * (b-a) + a
    
#     # get the pdf of the monte carlo elements
#     cdef np.ndarray[double, ndim=1] pdf_mc = pdf(X_mc, **args)
    
#     # computes the volume of the integer
#     cdef double V = np.prod((b-a))
    
#     # initialize the array to return
#     cdef np.ndarray[double, ndim=1] cdf = calcul(X, X_mc, pdf_mc)
    
#     cdf *= V / n_mc
    
#     return(cdf)

# cpdef np.ndarray[double, ndim=2] _generate(int size,
#                                            int n_features):
    
#     cdef np.ndarray[double, ndim=2] X_mc = np.random.random(size=(size, n_features))
#     return(X_mc)

def calcul(np.ndarray[DTYPE_t, ndim=2] X,
            np.ndarray[DTYPE_t, ndim=2] X_mc,
            np.ndarray[DTYPE_t, ndim=1] pdf_mc):
    
    assert X.dtype == DTYPE and X_mc.dtype == DTYPE and pdf_mc.dtype == DTYPE
    
    # the number of elements
    cdef long n_samples = X.shape[0]
    cdef long n_mc = X_mc.shape[0]
    
    cdef np.ndarray[DTYPE_t, ndim=1] cdf = np.zeros(n_samples, dtype=DTYPE)
    
    cdef long i
    for i in range(n_samples):
        # cdf[i] = pdf_mc[np.all(X_mc <= X[i,:], axis=1)].sum()
        cdf[i] = pdf_mc[0]
        # for j in range(n_mc):
            # cdf[i] = cdf[i] + pdf_mc[j]
    return(cdf)