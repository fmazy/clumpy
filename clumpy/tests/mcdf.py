#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 18:30:30 2021

@author: frem
"""

import sys
import numpy as np
import time

import sys
sys.path.insert(0,'../../')
sys.path.insert(0,'../../../multistats')
import clumpy
import multistats

#%%
import pymc3 as pm
with pm.Model() as model:
    parameter = pm.Exponential("poisson_param", 1.0)
    data_generator = pm.Poisson("data_generator", parameter)

#%%
n_features = 2
n_peaks = 1

def generate_mu_sigma_a_b(n_features, n_peaks):
    
    mu_ = []
    sigma_ = []
    
    a_ = np.zeros(n_features) + 1000
    b_ = np.zeros(n_features) - 1000
    
    for p in range(n_peaks):
        mu = np.random.random(n_features)
        sigma = np.diag(np.random.random(n_features))
        rho = np.random.random(int(n_features*(n_features-1) / 2)) * 0.8
        
        i_rho = 0
        for i in range(n_features):
            if n_features%2 == 0:
                n_columns = int(n_features / 2)
            else:
                n_columns = int(n_features / 2) + 1
            
            for j in range(n_columns):
                if i != j:
                    sigma[i,j] = rho[i_rho] * np.sqrt(sigma[i,i]*sigma[j,j])
                    sigma[j,i] = rho[i_rho] * np.sqrt(sigma[i,i]*sigma[j,j])
            
            a_[i] = np.min([a_[i], -6 * sigma[i,i] + mu[i]])
            b_[i] = np.max([b_[i],  6 * sigma[i,i] + mu[i]])
            
            
        mu_.append(mu)
        sigma_.append(sigma)
    
    return(mu_, sigma_, a_, b_)

mu, sigma, a, b = generate_mu_sigma_a_b(n_features, n_peaks)

#%%
from scipy.stats import multivariate_normal
def f_pdf(X):
    pdf = np.zeros(X.shape[0])
    for i in range(len(mu)):
        pdf += multivariate_normal.pdf(X, mean=mu[i], cov=sigma[i])
    
    pdf /= len(mu)    
    
    return(pdf)

def pdf_to_rvs(pdf, size, a, b, c, args):
    X = np.random.random((size, a.size)) * (b-a) + a
    C = np.random.random(size) * c
    pdf_X = pdf(X, **args)
    
    X = X[pdf_X>C,:]
    
    return(X)

#%%
def f2(x):
    return(f_pdf(np.array(x)[:,None])[0])

# def f2(x):
#     if x[0]>=0 and x[0]<=1 and x[1]>=0 and x[1]<=1:
#         return(1)
#     else:
#         return(0)

#%%
import vegas
integ = vegas.Integrator([[a[0], b[0]], [a[1],b[1]]])

result = integ(f2, nitn=10, neval=1000)
# print(result.summary())
# print('result = %s    Q = %.2f' % (result, result.Q))

#%%
I = multistats.integer(pdf=f_pdf,
                       a=a,
                       b=b)
print(I)
#%%
n_samples = 10000
X = np.random.random((n_samples, n_features)) * (b-a) + a

# f_X = f_pdf(X, mean=mu, cov=sigma)
# c = f_X.max() + f_X.var()

#%%
# n = 100000
# X = pdf_to_rvs(pdf=f_pdf, size=n, a=a, b=b, c=c, args={'mean':mu, 'cov':sigma})

#%%
# start_time = time.time()
# d1 = clumpy.metrics.pdf_to_mcdf(f_pdf,
#                                X[:,0],
#                                0,
#                                a,
#                                b,
#                                1000,
#                                {'mean':mu,
#                                 'cov':sigma})

# end_time = time.time()
# print(end_time-start_time)

# #%%
# start_time = time.time()
# d2 = multistats.pdf_to_mcdf(pdf=f_pdf,
#                             X=X[:,0],
#                             columns=0,
#                             a=a,
#                             b=b,
#                             n_mc=1000,
#                             method='monte_carlo',
#                             args={'mean':mu,
#                              'cov':sigma})

# end_time = time.time()
# print(end_time-start_time)

# #%%
# print(np.max(np.abs(d1-d2)))

#%%
column = 0
n_mc = 1000

X_prime = X.copy()
b_columns = np.arange(n_features)
b_columns = np.delete(b_columns, column)
X_prime[:,b_columns] = b[b_columns]

X_mc = multistats._monte_carlo._generate((n_mc, n_features))
pdf_mc = f_pdf(X_mc)

#%%
start_time = time.time()
cdf1 = np.zeros(n_samples)
# for each element
for i in range(n_samples):
    # sum all pdf for monte carlo elements lower or equal to X[i,:] 
    # for each features
    cdf1[i] = pdf_mc[np.all(X_mc <= X[i,:], axis=1)].sum()
    # cdf1[i] = 42.0
end_time = time.time()
print(end_time-start_time)

#%%
integ = vegas.Integrator([[a[0], b[0]], [a[1],b[1]]])

cdf2 = np.zeros(n_samples)
for i in range(n_samples):
    def f2(x):
        if np.all(x <= X[i,:]):
            return(f_pdf(np.array(x)[:,None])[0])
        else:
            return(0)
    
    cdf2[i] = integ(f2, nitn=1, neval=1000).mean
    

#%%
start_time = time.time()
cdf2 = clumpy.metrics.calcul(X, X_mc, pdf_mc)
end_time = time.time()
print(end_time-start_time)
