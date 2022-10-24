# -*- coding: utf-8 -*-

import numpy as np
from ekde import KDE
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import warnings
from matplotlib import pyplot as plt

from itertools import combinations
from os import sys
from ..tools._data import generate_grid

class CramerMRMR():
    """
    approx in  {'mean', 'median', 'std'}
    """
    def __init__(self, 
                 V_gof_min = 0.1, 
                 V_toi_max = 0.2,
                 epsilon=0.1,
                 alpha=0.9,
                 approx='mean',
                 kde_method=True,
                 kde_params={},
                 ev_labels=None,
                 k_shift=0,
                 region_label=None,
                 initial_state=None,
                 final_state=None):
        
        self.V_gof_min = V_gof_min
        self.V_toi_max = V_toi_max
        self.epsilon = epsilon
        self.alpha = alpha
        self.approx = approx   
        self.kde_method = kde_method
        self.kde_params = kde_params
        self.ev_labels = ev_labels
        self.k_shift = k_shift
        self.region_label=region_label,
        self.initial_state=initial_state,
        self.final_state=final_state
        
        super().__init__()
    
    def __repr__(self):
        return 'CramerMRMR()'
    
    def set_params(self, **params):
        """
        Set parameters.

        Parameters
        ----------
        **params : kwargs
            Parameters et values to set.

        Returns
        -------
        self : CramerMRMR
            The self object.

        """
        for param, value in params.items():
            setattr(self, param, value)
    
    def fit(self, Z, transited_pixels, bounds=None):
        self._cols_support = self.mrmr_cramer(Z, transited_pixels, bounds)
        
        return(self)
    
    def gof_Z(self, z, transited_pixels, k, bound='none'):
        self.compute_bins_1d(z, transited_pixels, k)
        if self.kde_method:
            df = self.get_gof_df_kde(z=z, 
                                     transited_pixels=transited_pixels,
                                     k=k, 
                                     bound=bound)
            
        else:
            df = self.get_gof_df(z, transited_pixels, k)
            
        return(self.gof(df, k))

    def compute_bins_1d(self, z, transited_pixels, k):
        n = transited_pixels.sum()
        
        n_m = int(np.max((n / (1 + n * self.epsilon**2),5)))
        
        Delta = np.max(z[transited_pixels]) - np.min(z[transited_pixels])
        
        if self.approx == 'mean':
            Gamma = np.max((10, int(n/n_m)))
            
        elif self.approx == 'median':
            kde = KDE()
            kde.set_params(**self.kde_params)
            kde.fit(z[:,None][transited_pixels])
            y = np.linspace(np.min(z), np.max(z), int(Delta / (kde._h / kde.q)))
            rho = kde.predict(y[:,None])
            Gamma = np.max((10, int(n * np.max(rho) * Delta / 2 / n_m)))
            
        elif self.approx == 'std':
            Gamma = np.max((10, int(n * Delta / n_m / 2 / np.std(z))))
            
        else:
            raise(ValueError("Unexpected 'self.approx' attribute value. It should be 'median' (default), 'mean', or 'std'"))
        
        
        if Gamma < 10:
            raise(ValueError("The number of bins (Gamma) is too low. Increase the epsilon parameter."))
        
        # print('digitizing, n_m=',n_m, ' Gamma=',Gamma)
        
        delta = Delta / Gamma
        self._1d_bins[k] = np.linspace(np.min(z[transited_pixels]), 
                           np.max(z[transited_pixels])+10**-5*delta, 
                           Gamma)

    def get_gof_df(self, z, transited_pixels, k):
        bins = self._1d_bins[k]
        g = np.digitize(z, bins)
        
        width = np.diff(bins)
        bins_centers = bins[:-1] + width / 2
        
        df = pd.DataFrame(bins_centers, columns=['z'])
        df['g'] = np.arange(df.index.size) + 1
        df['width'] = width
        
        G = pd.DataFrame(g, columns=['g'])
        df_O = G.loc[transited_pixels].groupby('g').size().reset_index(name='O')
        df_E = G.groupby('g').size().reset_index(name='E')
        
        df = df.merge(df_O, how='left')
        df = df.merge(df_E, how='left')
        df.fillna(0, inplace=True)
        
        df['E'] = df['E'] / df['E'].sum() * df['O'].sum()
        
        return(df)
    
    def get_gof_df_kde(self, z, transited_pixels, k, bound='none'):
        
        n = transited_pixels.sum()
        
        bins = self._1d_bins[k]
        bins_centers = bins[:-1] + np.diff(bins) / 2
        
        if bound is None:
            bound = 'none'
        
        if bound == 'none':
            bounds = []
        else:
            bounds = [(0, bound)]
            
        kde_O = KDE(bounds=bounds)
        kde_O.set_params(**self.kde_params)
        kde_O.fit(z[transited_pixels][:,None])
        O = kde_O.predict(bins_centers[:,None])
        O = O / np.sum(O) * n
        
        kde_E = KDE(bounds=bounds)
        kde_E.set_params(**self.kde_params)
        kde_E.fit(z[:,None])
        E = kde_E.predict(bins_centers[:,None]) * n
        E = E / np.sum(E) * n
        
        df = pd.DataFrame(bins_centers, columns=['z'])
        df['width'] = np.diff(bins)
        df['O'] = O
        df['E'] = E
        
        return(df)

    def gof(self, df, k): 
        
        n = df['O'].sum()
        
        if n == 0:
            return(np.nan)
        
        n_m = int(np.max((n / (1 + n * self.epsilon**2),5)))
        
        # df['keep'] = df['O'] >= n_m
        df['keep'] = df['E']>=5
        excluded = df.loc[~df['keep'], 'O'].sum() / n
        
        # recompute
        Gamma = df.loc[df['keep']].index.size
        n = df.loc[df['keep'], 'O'].sum()
        # print('recompute Gamma=',Gamma, ' excluded=',round(excluded,4)*100,'%')
                        
        # R
        R_mean = n / Gamma / n_m
        R_max = df.loc[df['keep'], 'O'].max() / n_m
        
        # rescale
        df['E'] = df['E'] / df.loc[df['keep'], 'E'].sum() * n
        
        # chi2
        chi2 = ((df.loc[df['keep'], 'O'] - df.loc[df['keep'], 'E'])**2 / df.loc[df['keep'], 'E']).sum()
        
        # Normality ratio
        df['R'] = (df['O'] - df['E']) / ( df['E']**0.5 )
        R_mean = df.loc[df['keep'], 'O'].sum() / Gamma / n_m
        R_max = df.loc[df['keep'], 'O'].max() / n_m
        # R_mean = df.loc[df['keep'], 'R'].abs().mean()
        # R_max = df.loc[df['keep'], 'R'].abs().max()
        
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
    
    def toi_Z(self, Z, k0, k1, bounds):
        
        self.compute_bins_2d(Z, k0, k1)
        
        
        if self.kde_method:
            df = self.get_toi_df_kde(Z=Z, 
                                     k0=k0, 
                                     k1=k1,
                                     bounds=bounds)
            
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
                kde = KDE()
                kde.set_params(**self.kde_params)
                kde.fit(z[:,None])
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
        
        # print('digitizing, n_m=',n_m,' Gamma=',Gamma)
    
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
    
    def get_toi_df_kde(self, Z, k0, k1, bounds=['none','none']):
        bins = self._2d_bins[(k0,k1)]
        
        widths = [np.diff(binsk) for binsk in bins]
        
        bins_centers = [bins[k][:-1] + widths[k] / 2 for k in range(2)]
        grid = generate_grid(*bins_centers)
        
        n, d = Z.shape
        
        if bounds is None:
            bounds = []
        
        new_bounds = []
        bounds_N0 = []
        bounds_N1 = []
        for i, bound in enumerate(bounds):
            if bound in ['left', 'right', 'both']:
                new_bounds.append((i, bound))
                if i == 0:
                    bounds_N0 = [(0, bound)]
                elif i == 1:
                    bounds_N1 = [(0, bound)]
                    
        bounds = new_bounds
        
        kde_O = KDE(bounds=bounds)
        kde_O.set_params(**self.kde_params)
        kde_O.fit(Z)
        O = kde_O.predict(grid)
        O = O / np.sum(O) * n
        
        kde_N0 = KDE(bounds=bounds_N0)
        kde_N0.set_params(**self.kde_params)
        kde_N0.fit(Z[:,[0]])
        N0 = kde_N0.predict(grid[:,[0]])
        N0 = N0 / N0.sum() * n
        
        kde_N1 = KDE(bounds=bounds_N1)
        kde_N1.set_params(**self.kde_params)
        kde_N1.fit(Z[:,[1]])
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
        # df['keep'] = df['O']>=n_m
        df['keep'] = df['E']>=5
        excluded = df.loc[~df['keep'], 'O'].sum() / n
        # df = df.loc[df['keep']]
        
        # recompute
        n = df.loc[df['keep'], 'O'].sum()
        Gamma = [len(np.unique(df.loc[df['keep'], 'g'+str(k)].values)) for k in [k0,k1]]
        Gamma01 = df.loc[df['keep']].index.size
        
        # print('recompute Gamma=', Gamma, ' unique :', Gamma01)
        
        if Gamma01 < 10:
            print('Warning, Gamma01 too low:', Gamma01)
        
        # rescale
        df['E'] = df['E'] / df.loc[df['keep'], 'E'].sum() * n
            
        # R
        df['R'] = (df['O'] - df['E']) / df['E']**0.5
        R_mean = df.loc[df['keep'], 'O'].sum() / Gamma[0] / Gamma[1] / n_m
        R_max = df.loc[df['keep'], 'O'].max() / n_m
                
        # chi2
        chi2 = ((df.loc[df['keep'], 'O'] - df.loc[df['keep'], 'E'])**2 / df.loc[df['keep'], 'E']).values.sum()
        
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
    
    def mrmr_cramer(self, Z, transited_pixels, bounds=None):
        
        n, d = Z.shape
        
        if bounds is None:
            bounds = ['none' for k in range(d)]
        
        if len(bounds) != d:
            raise(ValueError("Unexpected 'bounds' parameter. It should be a list of bounds for each explanatory variable or 'None'. A bound can be {'none', 'left', 'right', 'both'}."))
        
        self._n = n
        self._1d = {}
        self._1d_bins = {}
        self._V_gof = {}
        self._n_m_gof = {}
        self._Gamma_gof = {}
        self._R_mean_gof = {}
        self._R_max_gof = {}
        self._excluded_gof = {}
        
        # Computing GoF
        # print('GoF')
        V_gof = []
        for k in range(d):
            # print('k', k)
            
            # exclusion of flat EV
            if Z[:,k].std() == 0:
                V = -1
            else:
                V = self.gof_Z(z=Z[:,k], 
                               transited_pixels=transited_pixels,
                               k=k, 
                               bound=bounds[k])
            V_gof.append(V)
        V_gof = np.array(V_gof)
        
        # V_gof = np.array([self.gof_Z(z=Z[:,k], 
        #                              transited_pixels=transited_pixels,
        #                              k=k, 
        #                              bound=bounds[k]) for k in range(d)])
        
        evs = np.arange(d)[V_gof >= self.V_gof_min]
        # print('V_gof', np.round(V_gof,4))
        # self.display_gof_results()
        
        # print('keep', evs)
        evs = evs[np.argsort(V_gof[evs])[::-1]]
        # return(evs)
        list_k0_k1 = list(combinations(evs, 2))
        
        # print(list_k0_k1)
        # return(evs)
        
        # Computing ToI
        # print('ToI')
        
        self._2d = {}
        self._2d_bins = {}
        self._V_toi = {}
        self._n_m_toi = {}
        self._Gamma_toi = {}
        self._R_mean_toi = {}
        self._R_max_toi = {}
        self._excluded_toi = {}
        
        for k0, k1 in list_k0_k1:
            if k0 in evs and k1 in evs:
                # print('toi, (k0,k1)=', (k0,k1))
                V_toi = self.toi_Z(Z=Z[:,[k0,k1]][transited_pixels],
                                   k0=k0, 
                                   k1=k1, 
                                   bounds=[bounds[k0], bounds[k1]])
                                
                # print('V_toi('+str(k0)+','+str(k1)+')='+str(np.round(V_toi,4)))
                if V_toi >= self.V_toi_max:
                    k_to_remove = np.array([k0,k1])[np.argmin(V_gof[[k0,k1]])]
                    # print('rm', k_to_remove)
                    evs = np.delete(evs, list(evs).index(k_to_remove))
                
        return evs
    
    def results_table(self):
        print('Cramer MRMR Results')
        print('===================')
        print('Region : '+self.region_label)
        print('initial_state : '+str(self.initial_state))
        print('final_state : '+str(self.final_state))
        print('n : '+str())
        print()
        print('Test of Relevance, V_GoF_min='+str(self.V_gof_min))
        print('EV | V_GoF | R_mean | R_max | excl. pix. | kept')
        for k in self._V_gof.keys():
            kept = 'n'
            if self._V_gof[k] >= self.V_gof_min:
                kept = 'Y'
            
            print(str(k)+' - '+self.ev_labels[k]+' | '
                  +str(round(self._V_gof[k],4))+' | '
                  +str(round(self._R_mean_gof[k],4))+' | '
                  +str(round(self._R_max_gof[k],4))+' | '
                  +str(round(self._excluded_gof[k]*100,2))+'% | '
                  +kept)
        print('---------\n')
        
        print('Test of Redundancy, V_ToI_max='+str(self.V_toi_max))
        print('EV couple | V_ToI | R_mean | R_max | excl. pix. | EV removed')
        for (k1,k2) in self._V_toi.keys():
            removed = '/'
            if self._V_toi[(k1, k2)] >= self.V_toi_max:
                if self._V_gof[k1] >= self._V_gof[k2]:
                    removed = str(k2)
                else:
                    removed = str(k1)
            
            print('('+str(k1)+','+str(k2)+') | '
                  +str(round(self._V_toi[(k1,k2)],4))+' | '
                  +str(round(self._R_mean_toi[(k1,k2)],4))+' | '
                  +str(round(self._R_max_toi[(k1,k2)],4))+' | '
                  +str(round(self._excluded_toi[(k1,k2)]*100,2))+'% | '
                  +removed)
        
        print('---------\n')
        print('selected EVs :'+str(self._cols_support))
    
    def tex_table(self, feature_names):
        k_shift = self.k_shift
        tab = "\\begin{tabular}{Sl|Sc|Sc|Sc|Sc}\n"
        tab += "\\multicolumn{5}{Sc}{\\textsc{Test of Relevance}, $V_{GoF}^m="+str(self.V_gof_min)+"$}\\\ \n"
        tab += "EV & $V_{GoF}$ & $R_{mean}$ & $R_{max}$ & excl. pix. \\\ \n\\hline\n"
    	# 1 - Elevation & & & & \\
    	# 1 - Elevation & & & & \\
    	# 1 - Elevation & & & & \\
    	# \hline
    	# \hline
    	# couple & $V_{ToI}$ & $R_{mean}$ & $R_{max}$ & excl. pix. \\
    	# (1,2) & & & & \\
    # 
        for k in range(len(self._V_gof.values())):
            key = list(self._V_gof.keys())[k]
            V = np.round(self._V_gof[key],4)
            R_mean = np.round(self._R_mean_gof[key],4)
            R_max = np.round(self._R_max_gof[key],4)
            excl = np.round(self._excluded_gof[key]*100,2)
            
            if V < self.V_gof_min:
                tab += feature_names[k] + " & "
                tab += "$" + str(V) + "$ &"
                tab += "$" + str(R_mean) + "$ &"
                tab += "$" + str(R_max) + "$ &"
                tab += "$" + str(excl) + "\\%$ \\\ \n"
            else:
                tab += "\\textbf{"+feature_names[k] + "} & "
                tab += "$\\textbf{" + str(V) + "}$ &"
                tab += "$\\textbf{" + str(R_mean) + "}$ &"
                tab += "$\\textbf{" + str(R_max) + "}$ &"
                tab += "$\\textbf{" + str(excl) + "}\\%$ \\\ \n"
        tab += "\hline\n"
        
        if len(self._V_toi.keys()) > 0:
            tab += "\\multicolumn{5}{Sc}{\\textsc{Test of Redundancy}, $V_{ToI}^M="+str(self.V_toi_max)+"$}\\\ \n"
            tab += "EV couple & $V_{ToI}$ & $R_{mean}$ & $R_{max}$ & excl. pix.\\\ \n"
            tab += "\\hline \n"
            for key in self._V_toi.keys():
                k0, k1 = key
                V = np.round(self._V_toi[key],4)
                R_mean = np.round(self._R_mean_toi[key],4)
                R_max = np.round(self._R_max_toi[key],4)
                excl = np.round(self._excluded_toi[key]*100,2)
                
                if V >= self.V_toi_max:
                    tab += "("+str(k0+k_shift)+","+str(k1+k_shift)+")"
                    if self._V_gof[k0] < self._V_gof[k1]: 
                        k_reject = k0
                    else:
                        k_reject = k1
                    tab += ", reject "+str(k_reject+k_shift)
                    tab += " & "
                    tab += "$" + str(V) + "$ &"
                    tab += "$" + str(R_mean) + "$ &"
                    tab += "$" + str(R_max) + "$ &"
                    tab += "$" + str(excl) + "\\%$ \\\ \n"
                    
                else:
                    tab += "\\textbf{("+str(k0+k_shift)+","+str(k1+k_shift)+")} & "
                    tab += "$\\textbf{" + str(V) + "}$ &"
                    tab += "$\\textbf{" + str(R_mean) + "}$ &"
                    tab += "$\\textbf{" + str(R_max) + "}$ &"
                    tab += "$\\textbf{" + str(excl) + "\\%}$ \\\ \n"
                
        tab += "\\hline \n"
        tab += "\\multicolumn{5}{Sc}{\\textsc{Selected EV : "
        for k in self._cols_support:
            tab += str(k+k_shift)
            if k != self._cols_support[-1]:
                tab += ", "
        tab += "}}\\\ \n"
        tab += "\\end{tabular}"
        return tab
    
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
    
    def subplot_gof(self, k):
        df = self._1d[k].copy()    
        
        # plt.bar(df['z'].values[:-1], 
        #         df['E'].values[:-1], 
        #         df['width'].values[1:], 
        #         align='center',
        #         color='tab:blue', 
        #         alpha=0.3,
        #         label='excluded $E_{\gamma_k}$')
        # plt.bar(df['z'].values[:-1], 
        #         df['O'].values[:-1], 
        #         df['width'].values[1:], 
        #         align='center',
        #         color='tab:orange', 
        #         alpha=0.3,
        #         label='excluded $O_{\gamma_k}$')
        
        plt.bar(df.loc[df['keep'], 'z'], 
                df.loc[df['keep'], 'E'], 
                df.loc[df['keep'], 'width'], 
                align='center',
                color='darkviolet', 
                alpha=0.7,
                label='$E_{\gamma_k}$')
        plt.bar(df.loc[df['keep'], 'z'], 
                df.loc[df['keep'], 'O'], 
                df.loc[df['keep'], 'width'],
                align='center',
                color='orange', 
                alpha=0.7,
                label='$O_{\gamma_k}$')
        
        # xlim = plt.xlim()
        
        # plt.plot(plt.xlim(),
        #          [cramer_mrmr._n_m_gof[k] for i in range(2)],
        #          ls='--',
        #          color='black',
        #          alpha=0.7,
        #          label='$n_m$',
        #          lw=2)
        
        # plt.xlim(xlim)
        
        xa, xb = plt.xlim()
        ya, yb = plt.ylim()
        
        plt.text(xa+(xb-xa)*0.5, ya+(yb-ya)*0.5, '$V_{GoF}='+
                  str(round(self._V_gof[k], 3))+
                  '$\n$R_{mean}='+str(round(self._R_mean_gof[k], 3))+'$\n$R_{max}='+str(round(self._R_max_gof[k], 3))+'$\nexluded pixels: '+str(round(self._excluded_gof[k]*100, 2))+'%')
        # plt.title('$V_{gof}='+
                  # str(round(cramer_mrmr._V_gof[v][k], 4))+
                  # '$')
        
        plt.legend()
        plt.grid()
        plt.xlabel(self.ev_labels[k])
        plt.ylabel('Number of Pixels')
        
    def plot_gof(self, path=None):
        plt.figure(figsize=(10,10))
        
        list_k = list(self._V_gof.keys())
        n_k = len(list_k)
        
        n_rows = int(n_k/2)
        if n_k%2>0:
            n_rows += 1
        
        for i, k in enumerate(list_k):
            plt.subplot(n_rows,2,i+1)
            self.subplot_gof(k)
        
        if path is not None:
            plt.savefig(path, bbox_inches='tight')
    
    
    def plot_toi(self, 
                 k0, k1, 
                 xlim=None, 
                 ylim=None, 
                 reorder=False, 
                 path=None,
                 cmap='plasma'):
        import matplotlib as mpl
        cmap = mpl.cm.get_cmap(cmap).copy()
        cmap.set_bad(color='#999')
        
        df = self._2d[(k0,k1)].copy()
        
        if reorder:
            df.sort_values(by='O', inplace=True, ascending=False)
            s = int(df.index.size**0.5)
            if s**2 == df.index.size:
                shape = (s, s)
            elif s * (s + 1) < df.index.size:
                shape = (s+1, s+1)
            else:
                shape = (s, s+1)
            grid = generate_grid(*[np.arange(s)+1 for s in shape])
            df[['g'+str(k0), 'g'+str(k1)]] = grid[:df.index.size]
            shape = (shape[0]+1, shape[0]+1)
        else:
            shape = (df['g'+str(k0)].values.max()+1, df['g'+str(k1)].values.max()+1)
            
        O = np.zeros(shape)
        O.fill(np.nan)
        E = np.zeros(shape)
        E.fill(np.nan)
        R = np.zeros(shape)
        R.fill(np.nan)
        
        for d in df.loc[df['keep']].iterrows():
            O[int(d[1]['g'+str(k0)]), int(d[1]['g'+str(k1)])] = d[1]['O']
            E[int(d[1]['g'+str(k0)]), int(d[1]['g'+str(k1)])] = d[1]['E']
            R[int(d[1]['g'+str(k0)]), int(d[1]['g'+str(k1)])] = d[1]['R']
        
        # df_out = df.loc[~df['keep']]
        
        plt.figure(figsize=(10,10))
        plt.subplot(2,2,1)
        im = plt.imshow(O.T, cmap=cmap)
        plt.colorbar(im,fraction=0.046, pad=0.04)
        # plt.scatter(df_out['g'+str(k0)], df_out['g'+str(k1)], marker='x', c='#ffffff')
        plt.title('$O_{\gamma_k}$')
        if xlim is None:
            plt.xlim([0.5,O.shape[0]-0.5])
        else:
            plt.xlim(xlim)
        if xlim is None:
            plt.ylim([0.5, O.shape[1]-0.5])
        else:
            plt.ylim(ylim)
        plt.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
        plt.xlabel(self.ev_labels[k0])
        plt.ylabel(self.ev_labels[k1])
        
        plt.subplot(2,2,2)
        im = plt.imshow(E.T, cmap=cmap)
        cb = plt.colorbar(im,fraction=0.046, pad=0.04)
        cb.set_label('Number of Pixels', labelpad=3)
        # plt.scatter(df_out['g'+str(k0)], df_out['g'+str(k1)], marker='x', c='#ffffff')
        plt.title('$E_{\gamma_k}$')
        if xlim is None:
            plt.xlim([0.5,O.shape[0]-0.5])
        else:
            plt.xlim(xlim)
        if xlim is None:
            plt.ylim([0.5, O.shape[1]-0.5])
        else:
            plt.ylim(ylim)
        plt.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)    
        plt.xlabel(self.ev_labels[k0])
        plt.ylabel(self.ev_labels[k1])
        
        plt.subplot(2,2,3)
        im = plt.imshow(np.abs(R.T), cmap=cmap)
        plt.colorbar(im,fraction=0.046, pad=0.04)
        
        # plt.scatter(df_out['g'+str(k0)], df_out['g'+str(k1)], marker='x', c='#ffffff')
        plt.title('$|R_{\gamma_k}|$')
        if xlim is None:
            plt.xlim([0.5,O.shape[0]-0.5])
        else:
            plt.xlim(xlim)
        if xlim is None:
            plt.ylim([0.5, O.shape[1]-0.5])
        else:
            plt.ylim(ylim)
        
        plt.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
        plt.xlabel(self.ev_labels[k0])
        plt.ylabel(self.ev_labels[k1])
        
        plt.subplot(2,2,4)
        df_sorted = df.sort_values(by='O', ascending=False)
        df_sorted.reset_index(inplace=True)
        # plt.bar(df_sorted.index, df_sorted['E'], align='edge', width=1, alpha=0.3, color='tab:blue', label='excluded $E_{\gamma_k}$')
        # plt.bar(df_sorted.index, df_sorted['O'], align='edge', width=1, alpha=0.3, color='tab:orange', label='excluded $O_{\gamma_k}$')
        plt.bar(df_sorted.loc[df_sorted['keep']].index, df_sorted.loc[df_sorted['keep'], 'E'], align='edge', width=1, alpha=0.8, color='darkviolet', label='$E_{\gamma_k}$')
        plt.bar(df_sorted.loc[df_sorted['keep']].index, df_sorted.loc[df_sorted['keep'], 'O'], align='edge', width=1, alpha=0.8, color='orange', label='$O_{\gamma_k}$')
        plt.grid()
        plt.xlabel('ordered 2d-bins')
        # plt.ylabel('Number of Pixels')
        
        xlim = plt.xlim()
        
        # plt.plot(plt.xlim(),
        #          [cramer_mrmr._n_m_toi[(k0,k1)] for i in range(2)],
        #          ls='--',
        #          color='black',
        #          alpha=0.7,
        #          label='$n_m$',
        #          lw=2)
        
        plt.xlim(xlim)
        
        xa, xb = plt.xlim()
        ya, yb = plt.ylim()
        
        plt.text(xa+(xb-xa)*0.5, ya+(yb-ya)*0.5, '$V_{ToI}='+
                  str(round(self._V_toi[(k0,k1)], 3))+
                  '$\n$R_{mean}='+str(round(self._R_mean_toi[(k0,k1)], 3))+'$\n$R_{max}='+str(round(self._R_max_toi[(k0,k1)], 3))+'$\nexluded pixels: '+str(round(self._excluded_toi[(k0,k1)]*100, 2))+'%')
        
        plt.legend()
        
        if path is not None:
            plt.savefig(path, bbox_inches='tight')
        
    def plot(self, path_prefix=None, extension='pdf'):
        
        if path_prefix is not None:
            path = path_prefix + '_gof.' + extension
        else:
            path = None
        
        self.plot_gof(path=path)
        
        for k0, k1 in self._V_toi.keys():
            if path_prefix is not None:
                path = path_prefix + '_toi_' + str(k0+self.k_shift) + '_' + str(k1+self.k_shift) + '.' + extension
            else:
                path = None
            
            self.plot_toi(k0=k0, k1=k1, path=path)