# -*- coding: utf-8 -*-

from .. import definition

import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from optbinning import MulticlassOptimalBinning
    
def binning(case:definition.Case, params):
    """
    Binning according to a case and parameters.


    see `numpy documentation <https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges>`_ -- Default value : ``'auto'``.
    
    Parameters
    ----------
    case : definition.Case
        the studied case
    
    params : dict
        the binning parameters for each feature. It is a dictionary which associates for each couples (initial state, feature name) a method and other options if needed. See the example below.   
    
    Notes
    -----
    
    A new attribute is then available :
        
        ``case.alpha``
            The returned bins for each transition and each feature.
    
    Example
    -------
    The binning parameters can be defined as following::
        
        binning_parameters = {(3, 'dem'):{'method':'optbinning'},
                          (3, 'slope'):{'method':'optbinning'},
                          (3,'distance_to_2'):{'method':'numpy', 'bins':20}}
    
    """
    case.alpha = pd.DataFrame(columns=['vi', 'Zk_name', 'alpha'])
    
    for (vi, feature_name), param in params.items():
        Zk = case.transitions.Ti[vi].Z[feature_name]
        alpha_sub = pd.DataFrame(columns=['vi', 'Zk_name', 'alpha'])
        
        if param['method'] == 'numpy':
            alpha_sub.alpha = _compute_bins_with_numpy(case.J, Zk, param['bins'])
            
        elif param['method'] == 'optbinning':
            alpha_sub.alpha = _compute_bins_with_optbinning(case.J, Zk)
            
        alpha_sub.vi = vi
        alpha_sub.Zk_name = Zk.name            
        case.alpha = pd.concat([case.alpha, alpha_sub], ignore_index=True)
            
def discretize(case:definition.Case, alpha=None):
    """
    Discretize the case.
    
    Parameters
    ----------
    case : definition.Case
        The case to discretize according to alpha
    alpha : Pandas DataFrame (default=None)
        The binning DataFrame. If ``None``, the self binning DataFrame computed by `binning` is used.
    
    Notes
    -----
    A new attribute is then available :
        
        ``case.discrete_J``
            The discretized features according to each studied pixels.
    
    """
    
    if type(alpha) == type(None):
        alpha = case.alpha
    else:
        case.alpha = alpha
    
    case.discrete_J = case.J.copy()
    
    features_to_discretize = alpha[['vi','Zk_name']].drop_duplicates()
    
    for row in features_to_discretize.itertuples():
        vi = row.vi
        feature_name = row.Zk_name
        alpha_Zk = alpha.loc[(alpha.vi == vi) &
                                  (alpha.Zk_name == feature_name), 'alpha'].values.astype(float)
        case.discrete_J.loc[case.discrete_J.v.i == vi, ('z', feature_name)] = np.digitize(case.discrete_J.loc[case.discrete_J.v.i == vi, ('z', feature_name)],
                                                                       bins=alpha_Zk)
        
    # fill na with 0
    case.discrete_J.fillna(value=0, inplace=True)
    
def _compute_bins_with_numpy(J, Zk, bins, sound=0, plot=0):
    data = J.loc[J.v.i == Zk.Ti.vi, ('z', Zk.name)]
    alpha_N, alpha = np.histogram(data, bins=bins)
    # # on agrandie d'un chouilla le dernier bin pour inclure le bord supérieur
    alpha[-1] += (alpha[-1]-alpha[-2])*0.001
    
    plt.step(alpha, np.append(alpha_N,0), where='post')
    plt.title(Zk.name)
    plt.show()
    
    return(alpha)

def _compute_bins_with_optbinning(J, Zk, sound=0, plot=0):
    optb = MulticlassOptimalBinning(name=Zk.name, solver="cp")
    x = J.loc[J.v.i == Zk.Ti.vi, ('z', Zk.name)].values
    y = J.loc[J.v.i == Zk.Ti.vi, ('v', 'f')].values
    
    optb.fit(x, y)
    
    if optb.status != 'OPTIMAL':
        sound = 2
    if sound >= 1 or plot==1:
        print(Zk.Ti.vi, Zk.name)
        print(optb.status)
        binning_table = optb.binning_table
        
        
        # print(binning_table.quality_score())
        if sound == 1:
            print(binning_table.build())
        if plot == 1:
            binning_table.plot()
        
        if sound >= 2:
            binning_table.build()
            binning_table.analysis()
    
    alpha = np.array([x.min()] + list(optb.splits) + [x.max()*1.001])
    
    return(alpha)

def export_bins_as_dinamica(case, path):
    """
    Export bins as a Dinamica like file.

    Parameters
    ----------
    case : definition.Case
        The case to discretize according to alpha
    path : str
        Output file path.

    """
    columns = ['From*', 'To*', 'Variable*', 'Range_Lower_Limit*', 'Weight']
    dinamica_ranges = pd.DataFrame(columns=columns)
    
    for Ti in case.transitions.Ti.values():
        for Tif in Ti.Tif.values():
            for Zk in Ti.Z.values():
                df = pd.DataFrame(columns=columns)
                df['Range_Lower_Limit*'] = case.alpha.loc[(case.alpha.vi==Ti.vi) &
                                                     (case.alpha.Zk_name == Zk.name)].alpha.values
                df['From*'] = Ti.vi
                df['To*'] = Tif.vf
                df['Variable*'] = Zk.name
                df['Weight'] = 0
                
                dinamica_ranges = pd.concat([dinamica_ranges, df])
    
    # create folder if not exists
    folder_name = os.path.dirname(path)
    if not os.path.exists(folder_name) and folder_name!= '':
        os.makedirs(folder_name)
    
    dinamica_ranges.to_csv(path, index=False)  