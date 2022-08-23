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
                 approx='mean',
                 Gamma_max=100,
                 kde_method=False,
                 rescale_E=False):
        
        self.initial_state = initial_state
        self.V_gof_min = V_gof_min
        self.V_toi_max = V_toi_max
        self.epsilon = epsilon
        self.alpha = alpha
        self.approx = approx   
        self.Gamma_max = Gamma_max
        self.kde_method = kde_method
        self.rescale_E = rescale_E 
        
        super().__init__()
    
    def __repr__(self):
        return 'CramerMRMR()'
    
    def fit(self, Z, V, v):
                
        self._cols_support = self.mrmr_cramer(Z, V, v)
                
        return(self)
    
    def gof_Z(self, z, id_v, v, k):
        self.compute_bins_1d(z, id_v, v, k)
        if self.kde_method:
            df = self.get_gof_df_kde(z, id_v, v, k)
            
        else:
            df = self.get_gof_df(z, id_v, v, k)
            
        return(self.gof(df, v, k))

    def compute_bins_1d(self, z, id_v, v, k):
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
        
        Gamma = np.min((Gamma, self.Gamma_max))
        
        if Gamma < 10:
            raise(ValueError("The number of bins (Gamma) is too low. Increase the epsilon parameter."))
        
        print('digitizing, n_m=',n_m, ' Gamma=',Gamma)
        
        delta = Delta / Gamma
        self._1d_bins[k] = np.linspace(np.min(z[id_v]), 
                           np.max(z[id_v])+10**-5*delta, 
                           Gamma)

    def get_gof_df(self, z, id_v, v, k):
        bins = self._1d_bins[k]
        g = np.digitize(z, bins)
        
        width = np.diff(bins)
        bins_centers = bins[:-1] + width / 2
        
        df = pd.DataFrame(bins_centers, columns=['z'])
        df['g'] = np.arange(df.index.size) + 1
        df['width'] = width
        
        G = pd.DataFrame(g, columns=['g'])
        df_O = G.loc[id_v].groupby('g').size().reset_index(name='O')
        df_E = G.groupby('g').size().reset_index(name='E')
        
        df = df.merge(df_O, how='left')
        df = df.merge(df_E, how='left')
        df.fillna(0, inplace=True)
        
        df['E'] = df['E'] / df['E'].sum() * df['O'].sum()
        
        return(df)
    
    def get_gof_df_kde(self, z, id_v, v, k):
        
        n = id_v.sum()
        
        bins = self._1d_bins[k]
        bins_centers = bins[:-1] + np.diff(bins) / 2
        
        kde_O = KDE(kernel='gaussian').fit(z[id_v][:,None])
        O = kde_O.predict(bins_centers[:,None])
        O = O / np.sum(O) * n
        
        kde_E = KDE(kernel='gaussian').fit(z[:,None])
        E = kde_E.predict(bins_centers[:,None]) * n
        E = E / np.sum(E) * n
        
        df = pd.DataFrame(bins_centers, columns=['z'])
        df['width'] = np.diff(bins)
        df['O'] = O
        df['E'] = E
        
        return(df)

    def gof(self, df, v, k): 
        
        n = df['O'].sum()
        n_m = int(np.max((n / (1 + n * self.epsilon**2),5)))
        
        df['keep'] = df['O'] >= n_m
        excluded = df.loc[~df['keep'], 'O'].sum() / n
        
        # recompute
        Gamma = df.loc[df['keep']].index.size
        n = df.loc[df['keep'], 'O'].sum()
        print('recompute Gamma=',Gamma, ' excluded=',round(excluded,4)*100,'%')
                        
        # R
        R_mean = n / Gamma / n_m
        R_max = df.loc[df['keep'], 'O'].max() / n_m
        
        # scale
        if self.rescale_E:
            df['E'] = df['E'] / df.loc[df['keep'], 'E'].sum() * n
        
        # chi2
        chi2 = ((df.loc[df['keep'], 'O'] - df.loc[df['keep'], 'E'])**2 / df.loc[df['keep'], 'E']).sum()
        
        # Normality ratio
        df['R'] = (df['O'] - df['E']) / ( df['E']**0.5 )
        
        if Gamma - 1 <= 0:
            print('Warning, the bins are too small for this transition.\nEventually increase the epsilon parameter.')
            return np.nan
        
        V_gof = (chi2 / n / (Gamma - 1))**0.5
        
        self._1d[k] = df
        self._V_gof[k] = V_gof
        self._n_m_gof[k] = n_m
        self._Gamma_gof[k] = Gamma
        self._R_mean_gof[k] = R_mean
        self._R_max_gof[k] = R_max
        self._excluded_gof[k] = excluded
        
        return V_gof
    
    def toi_Z(self, Z, v, k0, k1):
        
        self.compute_bins_2d(Z, k0, k1)
        
        
        if self.kde_method:
            df = self.get_toi_df_kde(Z, k0, k1)
            
        else:
            df = self.get_toi_df(Z, k0, k1)
            
        return(self.toi(df, k0, k1))
        
    
    def compute_bins_2d(self, Z, k0, k1):
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
        
        self._2d_bins[(k0,k1)] = bins
        
        print('digitizing, n_m=',n_m,' Gamma=',Gamma)
    
    def get_toi_df(self, Z, k0, k1):
        G = np.vstack([np.digitize(z, bins=self._2d_bins[(k0,k1)][k]) for k, z in enumerate(Z.T)]).T
        
        n, d = G.shape
        
        if d != 2:
            raise(ValueError("G is expected to have exactly 2 columns."))
        
        # populate multivariate bins
        G_df = pd.DataFrame(G.astype(int), columns=['g' + str(k) for k in [k0,k1]])
        df = G_df.groupby(['g'+str(k) for k in [k0,k1]]).size().reset_index(name='O')
                
        # merging with product bins
        df_N = [G_df.groupby('g'+str(k)).size().reset_index(name='N'+str(k)) for k in [k0,k1]]
        for df_Nk in df_N:
            df = df.merge(right=df_Nk, how='left')
            
        # compute E
        df['E'] = df['N'+str(k0)] * df['N'+str(k1)] / n
        
        return df
    
    def get_toi_df_kde(self, Z, k0, k1):
        bins = self._2d_bins[(k0,k1)]
        
        widths = [np.diff(binsk) for binsk in bins]
        
        bins_centers = [bins[k][:-1] + widths[k] / 2 for k in range(2)]
        grid = generate_grid(*bins_centers)
        
        n, d = Z.shape
        
        kde_O = KDE(kernel='box').fit(Z)
        O = kde_O.predict(grid)
        O = O / np.sum(O) * n
        
        kde_N0 = KDE(kernel='box').fit(Z[:,[0]])
        N0 = kde_N0.predict(grid[:,[0]])
        N0 = N0 / N0.sum() * n
        
        kde_N1 = KDE(kernel='box').fit(Z[:,[1]])
        N1 = kde_N1.predict(grid[:,[1]])
        N1 = N1 / N1.sum() * n
        
        E = N0 * N1
        E = E / E.sum() * n
        
        df = pd.DataFrame(grid, columns=['z'+str(k0), 'z'+str(k1)])
        df['g'+str(k0)] = np.digitize(df['z'+str(k0)], bins[0])
        df['g'+str(k1)] = np.digitize(df['z'+str(k1)], bins[1])
        
        df['O'] = O
        df['N'+str(k0)] = N0
        df['N'+str(k1)] = N1
        df['E'] = E
        
        return df
    
    def toi(self, df, k0, k1):
        # compute condition
        n = df['O'].sum()
        n_m = int(np.max((n / (1 + n * self.epsilon**2),5)))
        
        # exclude pixels
        df['keep'] = df['O']>=n_m
        excluded = df.loc[~df['keep'], 'O'].sum() / n
        # df = df.loc[df['keep']]
        
        # recompute
        n = df.loc[df['keep'], 'O'].sum()
        Gamma = [len(np.unique(df.loc[df['keep'], 'g'+str(k)].values)) for k in [k0,k1]]
        Gamma01 = df.loc[df['keep']].index.size
        
        print('recompute Gamma=', Gamma, ' unique :', Gamma01)
        
        if Gamma01 < 10:
            print('Warning, Gamma01 too low:', Gamma01)
        
        if self.rescale_E:
            df['E'] = df['E'] / df.loc[df['keep'], 'E'].sum() * n
            
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
        
        self._V_toi[(k0,k1)] = V_toi
        self._2d[(k0,k1)] = df
        self._n_m_toi[(k0,k1)] = n_m
        self._Gamma_toi[(k0,k1)] = Gamma01
        self._R_mean_toi[(k0,k1)] = R_mean
        self._R_max_toi[(k0,k1)] = R_max
        self._excluded_toi[(k0,k1)] = excluded
        
        return V_toi
    
    def mrmr_cramer(self, Z, V, v):
        id_v = V == v
        
        n, d = Z.shape
        self.bins = {}
        
        self._1d = {}
        self._1d_bins = {}
        self._V_gof = {}
        self._n_m_gof = {}
        self._Gamma_gof = {}
        self._R_mean_gof = {}
        self._R_max_gof = {}
        self._excluded_gof = {}
        
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
        
        self._2d = {}
        self._2d_bins = {}
        self._V_toi = {}
        self._n_m_toi = {}
        self._Gamma_toi = {}
        self._R_mean_toi = {}
        self._R_max_toi = {}
        self._excluded_toi = {}
        
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
    
    # def global_mrmr_cramer(self, Z, V, initial_state):
    #     n, d = Z.shape
        
    #     list_v = np.unique(V)
    #     id_evs = np.zeros(d).astype(bool)
        
        
    #     self._1d = {}
    #     self._1d_bins = {}
    #     self._V_gof = {}
    #     self._n_m_gof = {}
    #     self._Gamma_gof = {}
    #     self._R_mean_gof = {}
    #     self._R_max_gof = {}
    #     self._excluded_gof = {}
    #     self._2d = {}
    #     self._2d_bins = {}
    #     self._V_toi = {}
    #     self._n_m_toi = {}
    #     self._Gamma_toi = {}
    #     self._R_mean_toi = {}
    #     self._R_max_toi = {}
    #     self._excluded_toi = {}
        
    #     print(Z.min(axis=0))
        
    #     for v in list_v:
    #         if v == initial_state:
    #             continue
            
    #         print('==============')
    #         print('v=',v)
    #         print('==============')
            
    #         evs  = self.mrmr_cramer(Z, V, v)
                        
    #         print('v=',v, 'evs=', evs)
            
    #         id_evs[evs] = True
        
    #     evs = np.arange(d)[id_evs]
    #     print('==========')
    #     print('EVS : ', evs)
    #     return evs