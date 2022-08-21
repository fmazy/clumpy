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

    def digitize_1d(self, z, id_v, v, k):
        n = id_v.sum()
        
        n_m = int(np.max((n / (1 + n * self.epsilon**2),5)))
        
        Delta = np.max(z[id_v]) - np.min(z[id_v])
        
        if self.approx == 'mean':
            Gamma = np.max((10, int(n/n_m)))
            
        elif self.approx == 'median':
            kde = KDE().fit(z[:,None][id_v])
            y = np.linspace(np.min(z), np.max(z), int(Delta / (kde._h / kde.q)))
            rho = kde.predict(y[:,None])
            Gamma = np.max((10, int(n * np.max(rho) * Delta / 2 / n_m)))
            
        elif self.approx == 'std':
            Gamma = np.max((10, int(n * Delta / n_m / 2 / np.std(z))))
            
        else:
            raise(ValueError("Unexpected 'self.approx' attribute value. It should be 'median' (default), 'mean', or 'std'"))
        
        Gamma = np.min((Gamma, 100))
        
        if Gamma < 10:
            raise(ValueError("The number of bins (Gamma) is too low. Increase the epsilon parameter."))
        
        print('digitizing, n_m=',n_m, ' Gamma=',Gamma)
        
        delta = Delta / Gamma
        self._1d_bins[v][k] = np.linspace(np.min(z[id_v]), 
                           np.max(z[id_v])+10**-5*delta, 
                           Gamma)
        
        g = np.digitize(z, self._1d_bins[v][k])
               
        return g

    def gof(self, g, id_v, v, k): 
        G_df = pd.DataFrame(g, columns=['g'])
        df = G_df.groupby('g').size().reset_index(name='E')
        df_O = G_df.loc[id_v].groupby('g').size().reset_index(name='O')
        
        
        # print('new Gamma', Gamma)
        n = df_O['O'].sum()
        
        n_m = int(np.max((n / (1 + n * self.epsilon**2),5)))
        
        # restrict to enough populated bins:
        df_O['keep'] = df_O['O'] >= n_m
        excluded = df_O.loc[~df_O['keep'], 'O'].sum() / n
        # df_O = df_O.loc[df_O['keep']]
        
        # recompute
        Gamma = df_O.loc[df_O['keep']].index.size
        n = df_O.loc[df_O['keep'], 'O'].sum()
        print('recompute Gamma=',Gamma, ' excluded=',round(excluded,4)*100,'%')
        
        # merge
        df = df_O.merge(right=df, how='left')
        df.fillna(0, inplace=True) # just in case but not necessary
        
        # R
        R_mean = n / Gamma / n_m
        R_max = df.loc[df['keep'], 'O'].max() / n_m
        
        # scale
        df['E'] = df['E'] / df.loc[df['keep'], 'E'].sum() * n
        
        # chi2
        chi2 = ((df.loc[df['keep'], 'O'] - df.loc[df['keep'], 'E'])**2 / df.loc[df['keep'], 'E']).sum()
        
        # Normality ratio
        df['R'] = (df['O'] - df['E']) / ( df['E']**0.5 )
        
        if Gamma - 1 <= 0:
            print('Warning, the bins are too small for this transition.\nEventually increase the epsilon parameter.')
            return np.nan
        
        V_gof = (chi2 / n / (Gamma - 1))**0.5
        
        self._1d[v][k] = df
        self._V_gof[v][k] = V_gof
        self._n_m_gof[v][k] = n_m
        self._Gamma_gof[v][k] = Gamma
        self._R_mean_gof[v][k] = R_mean
        self._R_max_gof[v][k] = R_max
        self._excluded_gof[v][k] = excluded
        
        return V_gof
    
    def gof_Z(self, z, id_v, v, k):
        g = self.digitize_1d(z, id_v, v, k)
        
        return(self.gof(g, id_v, v, k))
    
    def digitize_2d(self, Z, v, k0, k1):
        n, d = Z.shape
        Delta = Z.max(axis=0) - Z.min(axis=0)
        
        n_m = int(np.max((n / (1 + n * self.epsilon**2),5)))
        
        if self.approx == 'mean':
            Gamma = [np.max((10,int((n/n_m)**0.5))) for k in range(2)]
            
        elif self.approx == 'median':
            Gamma = []
            for k, z in enumerate(Z.T):
                kde = KDE().fit(z[:,None])
                y = np.linspace(np.min(z), np.max(z), int(Delta[k] / (kde._h / kde.q)))
                rho = kde.predict(y[:,None])
                Gamma.append(np.max((10,int((n / n_m)**0.5 / 2 * np.max(rho) * Delta[k]))))
                
        elif self.approx == 'std':
            Gamma = []
            for k, z in enumerate(Z.T):
                Gamma.append(np.max((10,int((n / n_m)**0.5 * Delta[k] / np.std(z) / 2))))
        
        else:
            raise(ValueError("Unexpected 'self.approx' attribute value. It should be 'median' (default), 'mean', or 'std'"))
        
        bins = []
        for k, z in enumerate(Z.T):
            delta = Delta[k] / Gamma[k]
            bins.append(np.linspace(np.min(z), np.max(z)+10**-5*delta, Gamma[k]))
        
        self._2d_bins[v][(k0,k1)] = bins
        
        G = np.vstack([np.digitize(z, bins=bins[k]) for k, z in enumerate(Z.T)]).T
        
        print('digitizing, n_m=',n_m,' Gamma=',Gamma)
        
        return(G)
    
    def toi(self, G, v, k0, k1):
        n, d = G.shape
        
        if d != 2:
            raise(ValueError("G is expected to have exactly 2 columns."))
        
        # populate multivariate bins
        G_df = pd.DataFrame(G.astype(int), columns=['g' + str(k) for k in range(2)])
        df = G_df.groupby(['g0', 'g1']).size().reset_index(name='O')
        
        # compute condition
        n = df['O'].sum()
        n_m = int(np.max((n / (1 + n * self.epsilon**2),5)))
        
        
        # exclude pixels
        df['keep'] = df['O']>=n_m
        excluded = df.loc[~df['keep'], 'O'].sum() / n
        # df = df.loc[df['keep']]
        
        # recompute
        n = df.loc[df['keep'], 'O'].sum()
        Gamma = [len(np.unique(df.loc[df['keep'], 'g'+str(k)].values)) for k in range(2)]
        Gamma01 = df.loc[df['keep']].index.size
        
        print('recompute Gamma=', Gamma, ' unique :', Gamma01)
        
        if Gamma01 < 10:
            print('Warning, Gamma01 too low:', Gamma01)
        
        # merging with product bins
        df_N = [G_df.groupby(['g'+str(k)]).size().reset_index(name='N'+str(k)) for k in [0,1]]
        for k in [0,1]:
            df = df.merge(right=df_N[k], how='left')
            
        # compute E
        df['E'] = df['N0'] * df['N1'] / n
        
        # R
        df['R'] = (df['O'] - df['E']) / df['E']**0.5
        R_mean = df.loc[df['keep'], 'O'].sum() / Gamma[0] / Gamma[1] / n_m
        R_max = df.loc[df['keep'], 'O'].max() / n_m
                
        # chi2
        chi2 = ((df.loc[df['keep'], 'O'] - df.loc[df['keep'], 'E'])**2 / df.loc[df['keep'], 'O']).values.sum()
        
        # gamma
        Gamma_min = np.min(Gamma)
        
        if Gamma_min - 1 <= 0:
            print('Warning, the bins are too small for this transition.\nEventually increase the epsilon parameter.')
            return np.nan
        
        V_toi = (chi2 / n / (Gamma_min - 1))**0.5
        
        self._V_toi[v][(k0,k1)] = V_toi
        self._2d[v][(k0,k1)] = df
        self._n_m_toi[v][(k0,k1)] = n_m
        self._Gamma_toi[v][(k0,k1)] = Gamma01
        self._R_mean_toi[v][(k0,k1)] = R_mean
        self._R_max_toi[v][(k0,k1)] = R_max
        self._excluded_toi[v][(k0,k1)] = excluded
        
        return V_toi
    
    def toi_Z(self, Z, v, k0, k1):
        G = self.digitize_2d(Z, v, k0, k1)
        
        return(self.toi(G, v, k0, k1))
    
    def mrmr_cramer(self, Z, V, v):
        id_v = V == v
        
        n, d = Z.shape
        self.bins = {}
        
        self._1d[v] = {}
        self._1d_bins[v] = {}
        self._V_gof[v] = {}
        self._n_m_gof[v] = {}
        self._Gamma_gof[v] = {}
        self._R_mean_gof[v] = {}
        self._R_max_gof[v] = {}
        self._excluded_gof[v] = {}
        
        print('Computing GoF')
        print('-------------')
        
        V_gof = np.array([self.gof_Z(Z[:,k1], id_v, v, k1) for k1 in range(d)])
        
        evs = np.arange(d)[V_gof >= self.V_gof_min]
        print('V_gof', np.round(V_gof,4))
        print('keep', evs)
        evs = evs[np.argsort(V_gof[evs])[::-1]]
        # return(evs)
        list_k1_k2 = list(combinations(evs, 2))
        
        # print(list_k1_k2)
        # return(evs)
        
        print('\nComputing ToI')
        print('-------------')
        
        self._2d[v] = {}
        self._2d_bins[v] = {}
        self._V_toi[v] = {}
        self._n_m_toi[v] = {}
        self._Gamma_toi[v] = {}
        self._R_mean_toi[v] = {}
        self._R_max_toi[v] = {}
        self._excluded_toi[v] = {}
        
        for k0, k1 in list_k1_k2:
            if k0 in evs and k1 in evs:
                print('toi, (k0,k1)=', (k0,k1))
                V_toi = self.toi_Z(Z[:,[k0,k1]][id_v], v, k0, k1)
                                
                print('V_toi('+str(k0)+','+str(k1)+')='+str(np.round(V_toi,4)))
                if V_toi >= self.V_toi_max:
                    k_to_remove = np.array([k0,k1])[np.argmin(V_gof[[k0,k1]])]
                    print('rm', k_to_remove)
                    evs = np.delete(evs, list(evs).index(k_to_remove))
                
        return evs
    
    def global_mrmr_cramer(self, Z, V, initial_state):
        n, d = Z.shape
        
        list_v = np.unique(V)
        id_evs = np.zeros(d).astype(bool)
        
        
        self._1d = {}
        self._1d_bins = {}
        self._V_gof = {}
        self._n_m_gof = {}
        self._Gamma_gof = {}
        self._R_mean_gof = {}
        self._R_max_gof = {}
        self._excluded_gof = {}
        self._2d = {}
        self._2d_bins = {}
        self._V_toi = {}
        self._n_m_toi = {}
        self._Gamma_toi = {}
        self._R_mean_toi = {}
        self._R_max_toi = {}
        self._excluded_toi = {}
        
        print(Z.min(axis=0))
        
        for v in list_v:
            if v == initial_state:
                continue
            
            print('==============')
            print('v=',v)
            print('==============')
            
            evs  = self.mrmr_cramer(Z, V, v)
                        
            print('v=',v, 'evs=', evs)
            
            id_evs[evs] = True
        
        evs = np.arange(d)[id_evs]
        print('==========')
        print('EVS : ', evs)
        return evs