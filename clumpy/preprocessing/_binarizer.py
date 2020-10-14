# -*- coding: utf-8 -*-

from .. import definition
from ..tools import np_suitable_integer_type

import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from optbinning import MulticlassOptimalBinning

class Binarizer():
    """
    Binarizer
    """
    def fit(self, case, fit_params, sound=0, plot=False):
        """
        Fit the binarizer.

        Parameters
        ----------
        J : pandas dataframe.
            A two level ``v`` and ``z`` (X) columns are expected. In case of using the optbinning method, ``vf`` is expected.
        fit_params : dict
            A dictionnary of parameters for each initial state and feature.
            
                - ``{'method':'optbinning'}`` : the optimal binning according to the final states.
                - ``{'method':'numpy', 'bins':'auto'}`` : the numpy binarization with its bins parameter method.
        sound : int, default=0
            If ``sound==0``, the method is quiet
        plot : boolean, default=False
            If ``True``, a plot is displayed.

        Examples
        --------
        >>> fit_params = {(1, 'z0'):{'method':'optbinning'},
              (1, 'z1'):{'method':'optbinning'}}
        >>> binarizer = clumpy.preprocessing.Binarizer()
        >>> binarizer.fit(J, fit_params, plot=True)

        """
        
        self.alpha = {}
    
        for (vi, feature_name), fit_param in fit_params.items(): 
            
            X = case.get_z(vi, feature_name)
            
            if sound>0:
                print(vi, feature_name)
            
            # if param['method'] == 'numpy':
            #     alpha_sub.alpha = _compute_bins_with_numpy(case.J, Zk, param['bins'])
                                
            if fit_param['method'] == 'numpy':
                if 'bins' not in fit_param.keys():
                    fit_param['bins'] = 'auto'
                self.alpha[(vi, feature_name)] = _compute_linear_bins(X, fit_param['bins'], sound=sound, plot=plot, plot_title='vi='+str(vi)+' - '+feature_name)
    
    def transform(self, case, inplace=False):
        """
        binarize the entry.

        Parameters
        ----------
        J : pandas dataframe.
            Data to transorm. A two level ``v`` and ``z`` (X) columns are expected.

        Returns
        -------
        A discretized pandas dataframe. Other columns are kept.

        """
        if not inplace:
            case = case.copy()

        for (vi, feature_name), alpha in self.alpha.items():
            
            column_id = case.get_z_column_id(vi, feature_name)
            
            case.Z[vi][:, column_id] = np.digitize(case.Z[vi][:, column_id],
                                                   bins=alpha)
            
        for vi in case.Z.keys():
            case.Z[vi] = np_suitable_integer_type(case.Z[vi])
    
        if not inplace:
            return(case)
    
    def inverse_transform(self, case, where='mean', inplace=False):
        """
        Inverse the binarization.

        Parameters
        ----------
        J : pandas dataframe.
            Data to inverse. A two level ``v`` and ``z`` (X) columns are expected.
        where : {'mean', 'right', 'left'} (default=``mean``)
            Where the value should be computed.
            
                - ``'mean'`` : the returned data is the averaged of bins limits.
                - ``'left'`` : the returned data is the low limit of the bin.
                - ``'right'`` : the returned data is the high limit of the bin.

        """
        if not inplace:
            case = case.copy()
        
        for (vi, feature_name), alpha in self.alpha.items():
            
            case.Z[vi] = case.Z[vi].astype(np.float)
            
            column_id = case.get_z_column_id(vi, feature_name)
            
            self._inverse_transform_vi(z=case.Z[vi][:, column_id],
                                       alpha=alpha,
                                       where=where)
            
            # idz = (case.Z[vi][:, column_id] != 0) & (case.Z[vi][:, column_id] != alpha.size)
            
            # if where=='mean':
            #     case.Z[vi][idz, column_id] = (alpha[case.Z[vi][idz, column_id].astype(np.int)-1] + alpha[case.Z[vi][idz, column_id].astype(np.int)]) / 2
                
            # elif where == 'left':
            #     case.Z[vi][idz, column_id] = alpha[case.Z[vi][idz, column_id].astype(np.int)-1]
                                                                                     
            # elif where == 'right':
            #     case.Z[vi][idz, column_id] = alpha[case.Z[vi][idz, column_id].astype(np.int)]
            
            # idz = case.Z[vi][:, column_id] == 0
            # case.Z[vi][idz, column_id] = - np.inf
            
            # idz = case.Z[vi][:, column_id] == alpha.size
            # case.Z[vi][idz, column_id] = np.inf
            
        if not inplace:
            return(case)
        
    def _inverse_transform_vi(self, z, alpha, where):
        
        idz = (z != 0) & (z != alpha.size)
        
        if where=='mean':
            z[idz] = (alpha[z[idz].astype(np.int)-1] + alpha[z[idz].astype(np.int)]) / 2
            
        elif where == 'left':
            z[idz] = alpha[z[idz].astype(np.int)-1]
                                                                                 
        elif where == 'right':
            z[idz] = alpha[z[idz].astype(np.int)]
        
        idz = z == 0
        z[idz] = - np.inf
        
        idz = z == alpha.size
        z[idz] = np.inf

def _compute_bins_with_optbinning(X, y, name='name', sound=0, plot=False):
    optb = MulticlassOptimalBinning(name=name, solver="cp")
    
    optb.fit(X, y)
    
    if optb.status != 'OPTIMAL':
        sound = 2
    if sound >= 1 or plot==True:
        if sound > 0:
            print(optb.status)
        binning_table = optb.binning_table
        
        
        # print(binning_table.quality_score())
        if sound == 1:
            print(binning_table.build())
        if plot:
            binning_table.build()
            binning_table.plot()
        
        if sound >= 2:
            binning_table.build()
            binning_table.analysis()
    
    alpha = np.array([X.min()] + list(optb.splits) + [X.max()*1.0001])
    
    return(alpha)

def _compute_linear_bins(X, bins, sound=0, plot=False, plot_title=''):
    alpha_N, alpha = np.histogram(X, bins=bins)
    # # on agrandie d'un chouilla le dernier bin pour inclure le bord supérieur
    alpha[-1] += (alpha[-1]-alpha[-2])*0.001
    
    if plot:
        plt.step(alpha, np.append(alpha_N,0), where='post')
        plt.title(plot_title)
        plt.show()
    
    return(alpha)

# def export_bins_as_dinamica(case, path):
#     """
#     Export bins as a Dinamica like file.

#     Parameters
#     ----------
#     case : definition.Case
#         The case to discretize according to alpha
#     path : str
#         Output file path.

#     """
#     columns = ['From*', 'To*', 'Variable*', 'Range_Lower_Limit*', 'Weight']
#     dinamica_ranges = pd.DataFrame(columns=columns)
    
#     for Ti in case.transitions.Ti.values():
#         for Tif in Ti.Tif.values():
#             for Zk in Ti.Z.values():
#                 df = pd.DataFrame(columns=columns)
#                 df['Range_Lower_Limit*'] = case.alpha.loc[(case.alpha.vi==Ti.vi) &
#                                                      (case.alpha.Zk_name == Zk.name)].alpha.values
#                 df['From*'] = Ti.vi
#                 df['To*'] = Tif.vf
#                 df['Variable*'] = Zk.name
#                 df['Weight'] = 0
                
#                 dinamica_ranges = pd.concat([dinamica_ranges, df])
    
#     # create folder if not exists
#     folder_name = os.path.dirname(path)
#     if not os.path.exists(folder_name) and folder_name!= '':
#         os.makedirs(folder_name)
    
#     dinamica_ranges.to_csv(path, index=False)  