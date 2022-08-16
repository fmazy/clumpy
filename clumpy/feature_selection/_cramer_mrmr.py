# -*- coding: utf-8 -*-

import numpy as np
from ._feature_selector import FeatureSelector
from ekde import KDE
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import warnings

from itertools import combinations
from os import sys
from ..tools._data import generate_grid

class CramerMRMR(FeatureSelector):
    """
    approx in  {'mean', 'median', 'std'}
    """
    def __init__(self, 
                 initial_state,
                 V_gof_min = 0.2, 
                 V_toi_max = 0.2,
                 epsilon=0.1,
                 alpha=0.9,
                 approx='mean'):
        
        self.initial_state = initial_state
        self.V_gof_min = V_gof_min
        self.V_toi_max = V_toi_max
        self.epsilon = epsilon
        self.alpha = alpha
        self.approx = approx   
        
        super().__init__()
    
    def __repr__(self):
        return 'CramerMRMR()'
    
    def _fit(self, X, y):
        
        self._cols_support = self.global_mrmr_cramer(Z=X, 
                                                     V=y, 
                                                     initial_state=self.initial_state)
        
        return(self)

    def gof(self, g, id_v, k):     
        N = g.size
        G_df = pd.DataFrame(g, columns=['g'])
        df = G_df.groupby('g').size().reset_index(name='E')
        df_O = G_df.loc[id_v].groupby('g').size().reset_index(name='O')
        
        df = df.merge(df_O, how='outer')
        
        df['O'] = df['O'] / df['O'].sum() * df['E'].sum()
        
        df.fillna(0, inplace=True)
        chi2 = ((df['O'] - df['E'])**2 / df['E']).values.sum()
        
        df['R'] = (df['O'] - df['E']) / ( df['E']**0.5 )
        
        self.df[k] = df
        
        return (chi2 / (N * (df.index.size - 1)/1))**0.5
        

    def toi(self, G):
        n, d = G.shape
        
        if d != 2:
            raise(ValueError("G is expected to have exactly 2 columns."))
        G_df = pd.DataFrame(G, columns=['g' + str(k) for k in range(2)])
        df = G_df.groupby(['g0', 'g1']).size().reset_index(name='O')
        
        N = df['O'].sum()
        
        n_crit23_u = np.max((n / (1 + n * self.epsilon**2),5))
        
        R_mean = df['O'].sum() / len(np.unique(df['g0'].values)) / len(np.unique(df['g1'].values)) / n_crit23_u
        R_max = df['O'].max() / n_crit23_u
        print('R_mean=', R_mean, 'R_max=', R_max)
        
        print('% of pixels excluded : '+str(np.round(df.loc[df['O']<n_crit23_u]['O'].sum() / N,4)*100)+'%')
        df = df.loc[df['O']>=n_crit23_u]
        N = df['O'].sum()
        # print('>>>', df.loc[df['O']<n_crit23_u]['O'].sum() / N)
        
        for k in [0,1]:
            df['N'+str(k)] = df.groupby(['g'+str(k)])['O'].transform('sum')
        
        
        df['E'] = df['N0'] * df['N1'] / N
                
        chi2 = ((df['O'] - df['E'])**2 / df['O']).values.sum()
        
        n_m = np.min([len(np.unique(df['g0'].values)),
                       len(np.unique(df['g1'].values))])
        
        
        
        if n_m - 1 <= 0:
            print('warning, this transition does not occur enough to be well calibrated.')
            return(1)
        
        return (chi2 / (N * (n_m - 1)))**0.5
    
    def digitize_1d(self, z, id_v, k):
        n = id_v.sum()
        
        n_m = np.max((n / (1 + n * self.epsilon**2),5))
        
        delta = np.max(z) - np.min(z)
        
        if self.approx == 'mean':
            n_bins = int(n/n_m)
            
        elif self.approx == 'median':
            kde = KDE().fit(z[:,None])
            y = np.linspace(np.min(z), np.max(z), int(delta / (kde._h / kde.q)))
            rho = kde.predict(y[:,None])
            n_bins = int(n * np.max(rho) * delta / 2 / n_m)
            
        elif self.approx == 'std':
            n_bins = int(n * delta / n_m / 2 / np.std(z))
            
        else:
            raise(ValueError("Unexpected 'self.approx' attribute value. It should be 'median' (default), 'mean', or 'std'"))
        
        R_mean = n / n_bins / n_m
        n_gamma = n / n_bins
        R_max = n_gamma / n_m
        
        print('k=',k, 'R_mean=', R_mean, 'R_max=', R_max)
        
        # print('n_bins=', n_bins)
        # n_bins = int(n/n_m)
        
        warnings.filterwarnings('ignore') 
        kbd = KBinsDiscretizer(n_bins=n_bins, 
                               strategy="quantile",
                               encode="ordinal")
        kbd.fit(z[id_v,None])
        g = kbd.transform(z[:,None])
        warnings.filterwarnings('default') 
        
        return g
    
    def gof_Z(self, z, id_v, k):
        g = self.digitize_1d(z, id_v, k)
        
        return(self.gof(g, id_v, k))
    
    def digitize_2d(self, Z):
        n, d = Z.shape
        
        n_m = int(np.max((n / (1 + n * self.epsilon**2),5)))
        
        if self.approx == 'mean':
            n_bins = int(np.sqrt(n/n_m))
            
        elif self.approx == 'median':
            n_bins = n / n_m / 4
            for z in Z.T:
                delta = z.max() - z.min()
                kde = KDE().fit(z[:,None])
                y = np.linspace(np.min(z), np.max(z), int(delta / (kde._h / kde.q)))
                rho = kde.predict(y[:,None])
                n_bins *= np.max(rho) * delta
            n_bins = int(n_bins)
                
        elif self.approx == 'std':
            n_bins = n / n_m / 4
            for z in Z.T:
                delta = z.max() - z.min()
                n_bins *= delta / np.std(z)
            n_bins = int(n_bins)
        
        else:
            raise(ValueError("Unexpected 'self.approx' attribute value. It should be 'median' (default), 'mean', or 'std'"))
        
        warnings.filterwarnings('ignore')
        kbd = KBinsDiscretizer(n_bins=n_bins, 
                               strategy="quantile",
                               encode="ordinal")
        G = kbd.fit_transform(Z)
        warnings.filterwarnings('default') 
        
        return(G)
    
    def toi_Z(self, Z):
        G = self.digitize_2d(Z)
        
        return(self.toi(G))
    
    def mrmr_cramer(self, Z, id_v):
        n, d = Z.shape
        self.bins = {}
        self.df = {}
        V_gof = np.array([self.gof_Z(Z[:,k1], id_v, k1) for k1 in range(d)])
        
        id_v_ref = np.zeros(id_v.size).astype(bool)
        id_v_ref[np.random.choice(n, id_v.sum())] = True
        print('averaged random variable GoF ', round(np.mean([self.gof_Z(Z[:,k1], id_v_ref, k1) for k1 in range(d)]),4))
        
        evs = np.arange(d)[V_gof >= self.V_gof_min]
        print('V_gof', np.round(V_gof,4))
        print('keep', evs)
        evs = evs[np.argsort(V_gof[evs])[::-1]]
        # return(evs)
        list_k1_k2 = list(combinations(evs, 2))
        
        # print(list_k1_k2)
        # return(evs)
        for k1, k2 in list_k1_k2:
            if k1 in evs and k2 in evs:
                V_toi = self.toi_Z(Z[:,[k1,k2]][id_v])
                print('V_toi('+str(k1)+','+str(k2)+')='+str(np.round(V_toi,4)))
                if V_toi >= self.V_toi_max:
                    k_to_remove = np.array([k1,k2])[np.argmin(V_gof[[k1,k2]])]
                    print('rm', k_to_remove)
                    evs = np.delete(evs, list(evs).index(k_to_remove))
        
        return(evs)
    
    def global_mrmr_cramer(self, Z, V, initial_state):
        n, d = Z.shape
        
        list_v = np.unique(V)
        id_evs = np.zeros(d).astype(bool)
        for v in list_v:
            if v == initial_state:
                continue
            
            print('==============')
            print('v=',v)
            print('==============')
            id_v = V == v
            evs = self.mrmr_cramer(Z, id_v)
            # 
            # selected = self.compute_selected(Z, id_v)
            # ddx = self.ddx
            # if ddx is None:
                # ddx = self.compute_ddx(Z, selected, id_v)
        
            # z_min = Z.min(axis=0)
            # z_max = Z.max(axis=0)+0.0001 * Z.std(axis=0)
            # dx = Z.std(axis=0)*ddx
            # bins = [np.arange(z_min[k], z_max[k], dx[k]) for k in range(d)]
                    
            # G = np.vstack([np.digitize(Z[:,k], bins[k]) for k in range(d)]).T
            # self.G = G
            # evs = self.mrmr_cramer(G, selected, id_v)
            
            print('v=',v, 'evs=', evs)
            
            id_evs[evs] = True
        
        evs = np.arange(d)[id_evs]
        print('==========')
        print('EVS : ', evs)
        return evs