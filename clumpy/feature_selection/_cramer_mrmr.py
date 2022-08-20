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

    def digitize_1d(self, z, id_v):
        n = id_v.sum()
        
        n_m = int(np.max((n / (1 + n * self.epsilon**2),5)))
        
        Delta = np.max(z) - np.min(z)
        
        if self.approx == 'mean':
            Gamma = int(n/n_m)
            
        elif self.approx == 'median':
            kde = KDE().fit(z[:,None])
            y = np.linspace(np.min(z), np.max(z), int(Delta / (kde._h / kde.q)))
            rho = kde.predict(y[:,None])
            Gamma = int(n * np.max(rho) * Delta / 2 / n_m)
            
        elif self.approx == 'std':
            Gamma = int(n * Delta / n_m / 2 / np.std(z))
            
        else:
            raise(ValueError("Unexpected 'self.approx' attribute value. It should be 'median' (default), 'mean', or 'std'"))
        
        if Gamma < 10:
            raise(ValueError("The number of bins (Gamma) is too low. Increase the epsilon parameter."))
        print('Gamma=',Gamma)
        kbd = KBinsDiscretizer(n_bins=Gamma, 
                               strategy="uniform",
                               encode="ordinal")
        kbd.fit(z[id_v,None])
        g = kbd.transform(z[:,None])
        
        self._1d_kbd[-1].append(kbd)
        
        return g

    def gof(self, g, id_v): 
        G_df = pd.DataFrame(g, columns=['g'])
        df = G_df.groupby('g').size().reset_index(name='E')
        df_O = G_df.loc[id_v].groupby('g').size().reset_index(name='O')
        
        
        # print('new Gamma', Gamma)
        n = df_O['O'].sum()
        
        n_m = np.max((n / (1 + n * self.epsilon**2),5))
        
        # restrict to enough populated bins:
        df_O['keep'] = df_O['O'] >= n_m
        excluded = df_O['keep'].mean()
        
        # df_O = df_O.loc[df_O['keep']]
        
        # recompute
        Gamma = df_O.loc[df_O['keep']].index.size
        print('new Gamma', Gamma)
        n = df_O.loc[df_O['keep'], 'O'].sum()
        
        # merge
        df = df_O.merge(right=df, how='left')
        df.fillna(0, inplace=True) # just in case but not necessary
        
        df.loc[~df['keep'], 'E'] = np.nan
        
        # R
        R_mean = n / Gamma / n_m
        R_max = df.loc[df['keep'], 'O'].max() / n_m
        
        # scale
        df['E'] = df['E'] / df['E'].sum() * n
        
        # chi2
        chi2 = ((df['O'] - df['E'])**2 / df['E']).sum()
        
        # Normality ratio
        df['R'] = (df['O'] - df['E']) / ( df['E']**0.5 )
        
        self._1d[-1].append(df)        
        
        V_gof = (chi2 / n / (Gamma - 1))**0.5
        
        return V_gof, R_mean, R_max, excluded
    
    def gof_Z(self, z, id_v):
        g = self.digitize_1d(z, id_v)
        
        return(self.gof(g, id_v))
    
    def digitize_2d(self, Z):
        n, d = Z.shape
        
        n_m = int(np.max((n / (1 + n * self.epsilon**2),5)))
        
        if self.approx == 'mean':
            Gamma = int((n/n_m)**0.5)
            
        elif self.approx == 'median':
            Gamma = []
            for z in Z.T:
                delta = z.max() - z.min()
                kde = KDE().fit(z[:,None])
                y = np.linspace(np.min(z), np.max(z), int(delta / (kde._h / kde.q)))
                rho = kde.predict(y[:,None])
                Gamma.append(int((n / n_m)**0.5 / 2 * np.max(rho) * delta))
                
        elif self.approx == 'std':
            Gamma = []
            for z in Z.T:
                delta = z.max() - z.min()
                Gamma.append(int((n / n_m)**0.5 * delta / np.std(z) / 2))
        
        else:
            raise(ValueError("Unexpected 'self.approx' attribute value. It should be 'median' (default), 'mean', or 'std'"))
        
        kbd = KBinsDiscretizer(n_bins=Gamma, 
                               strategy="uniform",
                               encode="ordinal")
        G = kbd.fit_transform(Z)
        
        self._2d_kbd[-1].append(kbd)
        
        return(G)
    
    def toi(self, G):
        n, d = G.shape
        
        if d != 2:
            raise(ValueError("G is expected to have exactly 2 columns."))
        
        G_df = pd.DataFrame(G.astype(int), columns=['g' + str(k) for k in range(2)])
        df = G_df.groupby(['g0', 'g1']).size().reset_index(name='O')
        
        n = df['O'].sum()
        
        n_m = int(np.max((n / (1 + n * self.epsilon**2),5)))
        
        R_mean = df['O'].sum() / len(np.unique(df['g0'].values)) / len(np.unique(df['g1'].values)) / n_m
        R_max = df['O'].max() / n_m
        
        df['keep'] = 0
        df.loc[df['O']>=n_m, 'keep'] = 1
        excluded = df['keep'].mean()
        self._2d[-1].append(df)
        
        df = df.loc[df['keep']==1]
        n = df['O'].sum()
        
        df_N = [G_df.groupby(['g'+str(k)]).size().reset_index(name='N'+str(k)) for k in [0,1]]
        for k in [0,1]:
            df = df.merge(right=df_N[k], how='left')
        
        df['E'] = df['N0'] * df['N1'] / n
                
        chi2 = ((df['O'] - df['E'])**2 / df['O']).values.sum()
        
        Gamma_min = np.min([len(np.unique(df['g0'].values)),
                       len(np.unique(df['g1'].values))])
        
        if Gamma_min - 1 <= 0:
            print('warning, this transition does not occur enough to be well calibrated.')
            return(1)
        
        V_toi = (chi2 / n / (Gamma_min - 1))**0.5
        
        return V_toi, R_mean, R_max, excluded
    
    def toi_Z(self, Z):
        G = self.digitize_2d(Z)
        
        return(self.toi(G))
    
    def mrmr_cramer(self, Z, id_v):
        n, d = Z.shape
        self.bins = {}
        
        R_mean_gof = {}
        R_max_gof = {}
        excluded_gof = {}
        
        V_gof = []
        for k1 in range(d):
            V_gof_k1, R_mean_gof[k1], R_max_gof[k1], excluded_gof[k1] = self.gof_Z(Z[:,k1], id_v)
            V_gof.append(V_gof_k1)
        
        V_gof = np.array(V_gof)
        
        id_v_ref = np.zeros(id_v.size).astype(bool)
        id_v_ref[np.random.choice(n, id_v.sum())] = True
        
        evs = np.arange(d)[V_gof >= self.V_gof_min]
        print('V_gof', np.round(V_gof,4))
        print('keep', evs)
        evs = evs[np.argsort(V_gof[evs])[::-1]]
        # return(evs)
        list_k1_k2 = list(combinations(evs, 2))
        
        # print(list_k1_k2)
        # return(evs)
        
        V_toi = {}
        R_mean_toi = {}
        R_max_toi = {}
        excluded_toi = {}
        
        for k1, k2 in list_k1_k2:
            if k1 in evs and k2 in evs:
                V_toi[(k1,k2)], R_mean_toi[(k1,k2)], R_max_toi[(k1,k2)], excluded_toi[(k1,k2)] = self.toi_Z(Z[:,[k1,k2]][id_v])
                
                print('V_toi('+str(k1)+','+str(k2)+')='+str(np.round(V_toi[(k1,k2)],4)))
                if V_toi[(k1,k2)] >= self.V_toi_max:
                    k_to_remove = np.array([k1,k2])[np.argmin(V_gof[[k1,k2]])]
                    print('rm', k_to_remove)
                    evs = np.delete(evs, list(evs).index(k_to_remove))
        
        V_gof = {k1:V_gof[k1] for k1 in range(d)}
        
        extra = {'V_gof':V_gof,
                 'R_mean':R_mean_gof,
                 'R_max':R_max_gof,
                 'excluded_gof':excluded_gof,
                 'V_toi':V_toi,
                 'R_mean_toi':R_mean_toi,
                 'R_max_toi':R_max_toi,
                 'excluded_toi':excluded_toi}
        
        
        return evs, extra
    
    def global_mrmr_cramer(self, Z, V, initial_state):
        n, d = Z.shape
        
        list_v = np.unique(V)
        id_evs = np.zeros(d).astype(bool)
        
        
        self._1d = []
        self._1d_kbd = []
        
        self._2d = []
        self._2d_kbd = []
        
        self._V_gof = {}
        self._R_mean_gof = {}
        self._R_max_gof = {}
        self._excluded_gof = {}
        
        self._V_toi = {}
        self._R_mean_toi = {}
        self._R_max_toi = {}
        self._excluded_toi = {}
        
                
        for v in list_v:
            if v == initial_state:
                continue
            
            print('==============')
            print('v=',v)
            print('==============')
            id_v = V == v
            
            for obj in [self._1d, 
                        self._1d_kbd,
                        self._2d,
                        self._2d_kbd]:
                obj.append([])
            
            evs, extra  = self.mrmr_cramer(Z, id_v)
            self._V_toi[v] = extra['V_toi']
            self._R_mean_toi[v] = extra['R_mean_toi']
            self._R_max_toi[v] = extra['R_max_toi']
            self._excluded_toi[v] = extra['excluded_toi']
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