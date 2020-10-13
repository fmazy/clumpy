# -*- coding: utf-8 -*-
"""
tools.py
====================================
The tool module of demeter
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def human_size(s):
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB']
    i = 0
    while round(s/1024) > 0:
        s = s / 1024
        i += 1
    
    return(s, units[i])

def np_suitable_integer_type(a):
    if a.min() >= 0: # if unsigned
        t = np.uint64
        m = a.max()
        if m <= 4294967295:
            t = np.uint32
            if m <= 65535:
                t = np.uint16
                if m<= 255:
                    t = np.uint8
    else:
        t = np.int64
        m = np.abs(a).max()
        if m <= 2147483647:
            t = np.uint32
            if m <= 32767:
                t = np.uint16
                if m<= 127:
                    t = np.uint8
                    
    return(a.astype(t))

def np_drop_duplicates_from_column(a, column_id):
    return(a[np.unique(a[:, column_id], return_index=True)[1]])
        
def plot_histogram(h, t='step', color=None, linestyle=None, linewidth=None, label=None, title=None, show=True):

    if t == 'step':
        y = np.append(h[0], 0)
        x = h[1]
        plt.step(x=x,
                y=y,
                where='post',
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label)
    
    elif t =='bar':
        height = h[0]
        x = h[1][:-1]
        
        plt.bar(x=x,
                height=height,
                width=np.diff(h[1])*0.9,
                align='edge',
                color=color,
                label=label)
    
    if title is not None:
        plt.title(title)
    
    if show:
        plt.show()

def flat_midf(df, inplace=False):
    if not inplace:
        df = df.copy()
        
    df.columns = df.columns.to_list()
    
    if not inplace:
        return(df)

def unflat_midf(df, inplace=False):
    if not inplace:
        df = df.copy()
    
    df.columns = pd.MultiIndex.from_tuples(df.columns.to_list())
    
    if not inplace:
        return(df)

def divideWith0(a, b, out=0):
    if out == 0:
        c = np.divide(a, b, out = np.zeros_like(a), where= b!=0)
    elif out == 'nan':
        c = np.divide(a, b, out = np.zeros_like(a).fill(np.nan), where= b!=0)
    return(c)

def df_to_matrix(df, shape, coord_names, x_name):
    """
    only 2 dimensions for now...
    """
        
    M = np.zeros(shape)
    
    df['flat_index'] = np.ravel_multi_index(df[coord_names].values.T, dims=shape)    
    # df = df.set_index([i_name, j_name])
    
    M.flat[df.flat_index.values] = df[x_name].values
    
    df.drop('flat_index', axis=1, inplace=True)
    
    return(M)
    
def matrix_to_df(M, coord_names, x_name, drop_null=True, reset_index=True, set_index=False):
    
    df = pd.DataFrame(columns=coord_names+[x_name])
    
    df[x_name] = M.flat
    coords = np.unravel_index(np.arange(M.size), np.shape(M))
    df[coord_names] = np.array(coords).T

    
    if drop_null:
        df = df.loc[df[x_name]!=0]
        
    if reset_index:
        df.reset_index(inplace=True, drop=True)

    if set_index:
        df.set_index(['vi','vf'], inplace=True)
    
    return(df)

# def df_matrix_to_df(df, i_name, j_name, x_name):
    # index = 

def generateBigNp(shape, index, values, default_value=0):
    M = np.zeros(shape)
    M.fill(default_value)
    M.flat[index] = values
    return(M)
    
def normalize(rawpoints, low=0.0, high=1.0):
    mins = np.min(rawpoints, axis=None)
    maxs = np.max(rawpoints, axis=None)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

def smoothRollingMean(y, box_pts):
    """
    
    """
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def smooth(y, parameters, keep_sum=True):
    """
    smoothing function
    :param y: series to smooth
    :type y: numpy array
    :param parameters: smoothing parameters. they are detailled below
    :type parameters: dict
    :param keep_sum: if `True`, the data sum is kept.
    :type keep_sum: bool
    
    :returns: smoothed y
    
    smoothing parameters :
    * `'rolling_mean'` (moving average), parameters:
        * `'n'` box points number
        
    examples::
        parameters={'algorithm': 'rolling_mean',
                    'n': 2}
    """
    sum_y = np.sum(y)
    if parameters['algorithm'] == 'rolling_mean':
        y = smoothRollingMean(y, parameters['n'])
    
    if keep_sum:
        y = y/np.sum(y)*sum_y
    
    return(y)

def chi2(N,E):
    return(np.sum(np.power(N-E,2)/E))    

def cramers_V(N,E):
    
    nb = np.sum(N+E)
    V = np.sqrt(chi2(N,E)/nb)
    
    return(V)

def l2_distance(N,E):
    return(np.sqrt(np.sum(np.power(N-E,2))))

def draw_in_histogram(bins, hist, shape):
    
    # x = np.random.choice(bins)
    
    return(df)
    #     N = Ti.T.patchesHist.N.loc[(Ti.T.patchesHist.vi==Ti.vi) &
    #                                  (Ti.T.patchesHist.vf==Tif.vf) &
    #                                  (Ti.T.patchesHist.isl_exp == isl_exp)].values
    #     N = N[:-1] # pour rappel, on lui a ajouté un 0 !
        
    #     S = Ti.T.patchesHist.S.loc[(Ti.T.patchesHist.vi==Ti.vi) &
    #                                  (Ti.T.patchesHist.vf==Tif.vf) &
    #                                  (Ti.T.patchesHist.isl_exp == isl_exp)].values
        
    #     x_s = np.random.random(n)
    #     q_s = np.digitize(x_s, bins=np.cumsum(N*np.diff(S)/(N*np.diff(S)).sum()))

    #     J_vi_kernels_stock.loc[J_vi_kernels_stock.vf == Tif.vf,'S'] = S[q_s].astype(int)
    
    

#%%
#M=np.arange(10*10)
#M = M.reshape((10,10))
#M.fill(0)
##print(M)
#coord = (5,5)
#x = 0
#y = 0
#M[coord[0], coord[1]] = 1
##print(M[coord[0]+x, coord[1]+y])
#
#m = 30
#
#h = (-1,0)
#d = (0,1)
#b = (1,0)
#g = (0,-1)
#
#l = []
#k = 1
#id_l = 0
#for i in range(m-1):
#    if id_l == len(l):
#        l = []
#        for j in range(k):
#            l.append(h)
#        for j in range(k):
#            l.append(d)
#        for j in range(k+1):
#            l.append(b)
#        for j in range(k+1):
#            l.append(g)
#        k += 2
#        id_l = 0
#    x += l[id_l][0]
#    y += l[id_l][1]
#    M[coord[0]+x, coord[1]+y] = i+2
#    
#    id_l += 1
#
#print(M)
    