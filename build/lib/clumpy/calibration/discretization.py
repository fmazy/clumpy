"""
intro
"""

# external libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from sklearn.preprocessing import KBinsDiscretizer
from optbinning import MulticlassOptimalBinning

def compute_bins(J, Zk, param, sound=0, plot=0):     
    """
    :math:`Z_k` discretization
    
    :param bins: bins parameter, see `numpy documentation <https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges>`_ -- Default value : ``'auto'``.
    
    example of Dinamica EGO param_discretization for Zk :
        
        T1.Z['EV0'].param_discretization = {'method':'DinamicaEGO', 'increment':1, 'min_delta':0.1, 'max_delta':0.8, 'tolerance_angle':30}
        
    example of optimal binning :
        
        Z.param_discretization = {'method':'optbinning'}
    """
    
    if param['method'] == 'numpy':
        data = J.loc[J.v.i == Zk.Ti.vi, ('z', Zk.name)]
        alpha_N, alpha = np.histogram(data, bins=param['bins'])
        # # on agrandie d'un chouilla le dernier bin pour inclure le bord supérieur
        alpha[-1] += (alpha[-1]-alpha[-2])*0.001
        
        plt.step(alpha, np.append(alpha_N,0), where='post')
        plt.title(Zk.name)
        plt.show()
        
        return(alpha)
            
    elif param['method'] == 'optbinning':
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
    
    elif param['method'] == 'DinamicaEGO':
        increment = Zk.param_discretization['increment']
        min_delta = Zk.param_discretization['min_delta']
        max_delta = Zk.param_discretization['max_delta']
        tolerance_angle = Zk.param_discretization['tolerance_angle']
        
        J_vi_zk = J[[('v','i'),('z', Zk.name)]].loc[J.v.i == Zk.Ti.vi].copy()
        J_vi_zk.sort_values(by=('z', Zk.name), inplace=True)
                
        a = [J[('z', Zk.name)].min() + i * increment for i in range((int((J[('z', Zk.name)].max() - np.floor(J[('z', Zk.name)].min()))/increment) + 2))]
        
        q = list(np.histogram(J[('z', Zk.name)].values, bins=a)[0])
        
        # plt.step(a,np.append(q,0), where='post')
        plt.plot(a[:-1],q)
        
        i = 0
        while a[i] < a[-1]:
            a_prime = a[i] + min_delta
            q_prime = J_vi_zk.loc[(a_prime <= J[('z', Zk.name)]) &
                                  (J[('z', Zk.name)] < a[i+1])].index.size
            
            print(a[i], a[i+1])
            print(q[i], q[i+1])
            
            print(a[i], a_prime, a[i+1])
            print(q[i]-q_prime, q_prime, q[i+1])
            
            a.insert(i+1,a_prime)
            q.insert(i+1,q_prime)
            
            break
        q = list(np.histogram(J[('z', Zk.name)].values, bins=a)[0])
        plt.plot(a[:-1],q)
        plt.ylim([0,800])
            # q_i_prime = 
        
    # elif Zk.param_discretization['method'] == 'Kmeans':
    #     discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')
    #     discretizer.fit()
    #     alpha.transform(X)
    #     print(alpha)
    #     # return()
            
            
            
        
        # print(J_vi_zk)
        # print(a)
        # print(q)
        # return [data.min(), data.max()]
        
    return(False)
    
def compute_bins_all(J, T):
    alpha = pd.DataFrame(columns=['vi', 'Zk_name', 'alpha'])
    
    for Ti in T.Ti.values():
        for Zk in Ti.Z.values():            
            alpha_sub = pd.DataFrame(columns=['vi', 'Zk_name', 'alpha'])
            alpha_sub.alpha = compute_bins(J, Zk)
            alpha_sub.vi = Ti.vi
            alpha_sub.Zk_name = Zk.name
        
            alpha = pd.concat([alpha, alpha_sub], ignore_index=True)
            
    return(alpha)


def discretize_all(J, T, alpha, inplace=False):
    
    if not inplace:
        J_hat = J.copy()
    
    for Ti in T.Ti.values():
        for Zk in Ti.Z.values():
            alpha_Zk = alpha.loc[(alpha.vi == Ti.vi) & (alpha.Zk_name == Zk.name), 'alpha'].values.astype(float)
            
            if inplace:
                J.loc[J.v.i == Zk.Ti.vi, ('z', Zk.name)] = np.digitize(J.loc[J.v.i == Zk.Ti.vi, ('z', Zk.name)],
                                                                     bins=alpha_Zk)
            else:
                J_hat.loc[J_hat.v.i == Zk.Ti.vi, ('z', Zk.name)] = np.digitize(J.loc[J.v.i == Zk.Ti.vi, ('z', Zk.name)],
                                                                     bins=alpha_Zk)
    
    if inplace:
        return(None)
    else:
        return(J_hat)


def alphaDinamicaExport(T, alpha, file_path):
    columns = ['From*', 'To*', 'Variable*', 'Range_Lower_Limit*', 'Weight']
    dinamica_ranges = pd.DataFrame(columns=columns)
    
    for Ti in T.Ti.values():
        for Tif in Ti.Tif.values():
            for Zk in Ti.Z.values():
                df = pd.DataFrame(columns=columns)
                df['Range_Lower_Limit*'] = alpha.loc[(alpha.vi==Ti.vi) &
                                                     (alpha.Zk_name == Zk.name)].alpha.values
                df['From*'] = Ti.vi
                df['To*'] = Tif.vf
                df['Variable*'] = Zk.name
                df['Weight'] = 0
                
                dinamica_ranges = pd.concat([dinamica_ranges, df])
                
    
    dinamica_ranges.to_csv(file_path, index=False)  