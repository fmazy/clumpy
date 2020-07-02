"""
intro
"""

import numpy as np
import pandas as pd
from scipy import ndimage

from .. import tools

def cramer_V(N_zk_vi_vf, N_zk_vi_vf_ref, vi, vf, Zk_name):
    
    N_zk_vi_vf = N_zk_vi_vf.loc[(N_zk_vi_vf.vi == vi) &
                                (N_zk_vi_vf.vf == vf) &
                                (N_zk_vi_vf.Zk_name == Zk_name)]
    
    N_zk_vi_vf_ref = N_zk_vi_vf_ref.loc[(N_zk_vi_vf_ref.vi == vi) &
                                (N_zk_vi_vf_ref.vf == vf) &
                                (N_zk_vi_vf_ref.Zk_name == Zk_name)]
    
    # 0 excluded
    
    
    V = tools.cramerV(N_zk_vi_vf.N_zk_vi_vf,N_zk_vi_vf_ref.N_zk_vi_vf)
    
    print(V)

def cramer_von_mises(N_zk_vi_vf, N_zk_vi_vf_ref, vi, vf, Zk_name):
    N_zk_vi_vf = N_zk_vi_vf.loc[(N_zk_vi_vf.vi == vi) &
                                (N_zk_vi_vf.vf == vf) &
                                (N_zk_vi_vf.Zk_name == Zk_name)]
    
    N_zk_vi_vf_ref = N_zk_vi_vf_ref.loc[(N_zk_vi_vf_ref.vi == vi) &
                                (N_zk_vi_vf_ref.vf == vf) &
                                (N_zk_vi_vf_ref.Zk_name == Zk_name)]
    N_zk_vi_vf_ref = N_zk_vi_vf_ref.rename(columns={'N_zk_vi_vf':'N_zk_vi_vf_ref'})
    
    N_zk_vi_vf = N_zk_vi_vf.merge(right=N_zk_vi_vf_ref,
                                  how='left')
    
    N_zk_vi_vf['F']  = np.cumsum(N_zk_vi_vf.N_zk_vi_vf/N_zk_vi_vf.N_zk_vi_vf.sum())
    
    N_zk_vi_vf['F0'] = np.cumsum(N_zk_vi_vf.N_zk_vi_vf_ref / N_zk_vi_vf.N_zk_vi_vf_ref.sum())
    
    d = np.sqrt(np.sum(np.power(N_zk_vi_vf.F - N_zk_vi_vf.F0,2)))
        
    print(d)
    return(d)
    
    

def powerdiscrepancy(N_zk_vi_vf, N_zk_vi_vf_ref, vi, vf, Zk_name):
    N_zk_vi_vf = N_zk_vi_vf.loc[(N_zk_vi_vf.vi == vi) &
                                (N_zk_vi_vf.vf == vf) &
                                (N_zk_vi_vf.Zk_name == Zk_name)].N_zk_vi_vf.values
    
    N_zk_vi_vf_ref = N_zk_vi_vf_ref.loc[(N_zk_vi_vf_ref.vi == vi) &
                                (N_zk_vi_vf_ref.vf == vf) &
                                (N_zk_vi_vf_ref.Zk_name == Zk_name)].N_zk_vi_vf.values
    
    # sum to 1
    N_zk_vi_vf /= N_zk_vi_vf.sum() 
    N_zk_vi_vf_ref /= N_zk_vi_vf_ref.sum()
    
    
    # 0 excluded
    D, p = gof.powerdiscrepancy(N_zk_vi_vf, N_zk_vi_vf_ref)
    return(D,p)

# statsmodels.stats.gof.powerdiscrepancy(observed, expected)