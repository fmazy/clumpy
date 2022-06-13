# -*- coding: utf-8 -*-

import numpy as np
from ._feature_selector import FeatureSelector
from ekde import KDE
import pandas as pd

from itertools import combinations
from os import sys
from ..tools._data import generate_grid

class CramerMRMR(FeatureSelector):
    def __init__(self, 
                 initial_state,
                 V_gof_min = 0.2, 
                 V_toi_max = 0.2,
                 epsilon=0.1,
                 alpha=0.9,
                 q=51,
                 std_step=0.05,
                 ddx=None):
        
        self.initial_state = initial_state
        self.V_gof_min = V_gof_min
        self.V_toi_max = V_toi_max
        self.epsilon = epsilon
        self.alpha = alpha
        self.q = q
        self.std_step = std_step
        self.ddx = ddx
        
        super().__init__()
    
    def __repr__(self):
        return 'CramerMRMR()'
    
    def _fit(self, X, y):
        
        self._cols_support = self.global_mrmr_cramer(Z=X, 
                                                     V=y, 
                                                     initial_state=self.initial_state)
        
        return(self)
    
    def select(self, Z):
        n, d = Z.shape
        
        z_min = Z.min(axis=0)
        z_max = Z.max(axis=0)+0.0001 * Z.std(axis=0)
        
        dx = (z_max - z_min) / self.q
        bins = [np.arange(z_min[k], z_max[k], dx[k]) for k in range(d)]
        
        X = generate_grid(*bins) + dx / 2
        
        kde = KDE().fit(Z)
        y = kde.predict(X)
        
        
        df = pd.DataFrame(X, columns=['x'+str(k) for k in range(d)])
        
        df['y'] = y
        
        for k in range(d):
            df['g'+str(k)] = np.digitize(df['x'+str(k)].values, bins[k])
                
        df.sort_values(by='y', ascending=False, inplace=True)
        
        df['cs'] = df['y'][::-1].cumsum() / df['y'].sum()
        
        df['selected'] = df['cs'] >= 1- self.alpha
        
        df = df[['g'+str(k) for k in range(d)] + ['selected']]
        
        G = np.vstack([np.digitize(Z[:,k], bins[k]) for k in range(d)]).T
        
        df_Z = pd.DataFrame(G, columns=['g'+str(k) for k in range(d)])
        
        df_Z = df_Z.merge(df, how='left', on=['g'+str(k) for k in range(d)])
        df_Z['selected'].fillna(False,inplace=True)
        return(df_Z['selected'].values)
    
    def compute_selected(self, Z, id_v):
        n, d = Z.shape
        selected = np.ones(Z.shape[0]).astype(bool)
        
        for k1 in range(d):
            # print('k1', k1)
            
            selected = np.all((selected,self.select(Z[:,[k1]])), axis=0)
            
            for k2 in range(k1+1):
                # print('(k1,k2)', (k1,k2))
                if k1 == k2:
                    selected[id_v] = np.all((selected[id_v], self.select(Z[:,[k1]][id_v])), axis=0)
                else:
                    selected[id_v] = np.all((selected[id_v],self.select(Z[:,[k1,k2]][id_v])), axis=0)
                
                # print(k1,k2,selected[id_v].sum())
        print('% of selected pixels', selected.mean())
        return(selected)
    
    def tests(self, G, selected):
        n, d = G.shape
        
        G_df = pd.DataFrame(G, columns=['g'+str(k) for k in range(d)])
        G_selected_df = G_df.loc[selected].copy()
        
        G_df = G_df.groupby(list(G_df.columns.to_list())).size().reset_index(name='n')
        G_df = G_df.merge(G_selected_df.drop_duplicates(), how='inner')
        
        n_prime = G_df.n.sum()
        
        # G_df['pi'] = G_df.n / n_prime
        
        # print('>', G_df.n.min())
        
        # test1 = G_df.index.size >= 10
        print(G_df)
        print('>>', G_df.n.min())
        # print(G_df.n.min(), n_prime / (1 + n_prime * self.epsilon**2))
        test2 = G_df.n.min() >= 5
        test3 = G_df.n.min() >= n_prime / (1 + n_prime * self.epsilon**2)
        # print((test1, test2, test3))
        # return(test3, G_df.n.min(), n_prime / (1 + n_prime * epsilon**2))
        # return(G_df.index.size)
        return(np.all((test2, test3)))
    
    def compute_ddx(self, Z, selected, id_v):
        n, d = Z.shape
        Z_std = Z.std(axis=0)
        z_min = Z.min(axis=0)
        z_max = Z.max(axis=0)+0.0001 * Z.std(axis=0)
        
        keep_while = True
        
        ddx = np.ones(d) * self.std_step
        
        
        while keep_while:
            sys.stdout.write('\033[2K\033[1G')
            print(ddx, end="\r")
            # print(ddx)
            
            dx = Z_std*ddx
            bins = [np.arange(z_min[k], z_max[k], dx[k]) for k in range(d)]
                    
            G = np.vstack([np.digitize(Z[:,k], bins[k]) for k in range(d)]).T
            
            keep_while = False
            
            for k1 in range(d):
                if ~self.tests(G[:,[k1]], selected):
                    ddx[k1] += self.std_step
                    keep_while = True
                    break
                
                print('!!!', selected[id_v].sum())
                if ~self.tests(G[:,[k1]][id_v], selected[id_v]):
                    ddx[k1] += self.std_step
                    keep_while = True
                    break
                            
        
        keep_while = True
        while keep_while:
            sys.stdout.write('\033[2K\033[1G')
            print(ddx, end="\r")
            dx = Z_std*ddx
            bins = [np.arange(z_min[k], z_max[k], dx[k]) for k in range(d)]
                    
            G = np.vstack([np.digitize(Z[:,k], bins[k]) for k in range(d)]).T
            
            keep_while = False
            
            list_k1_k2 = np.array(list(combinations(np.arange(d), 2)))
            list_k1_k2 = list_k1_k2[np.random.choice(list_k1_k2.shape[0], list_k1_k2.shape[0], replace=False)]
            
            for k1, k2 in list_k1_k2:
                if ~self.tests(G[:,[k1, k2]][id_v], selected[id_v]):
                    ddx[k1] += self.std_step
                    ddx[k2] += self.std_step
                    keep_while = True
                    break
        print()
        return(ddx)
    
    def df_count(self, G, selected, columns, name):
        n, d = G.shape
        
        G_df = pd.DataFrame(G, columns=columns)
        G_selected_df = G_df.loc[selected].copy()
        G_selected_df.drop_duplicates(inplace=True)
        G_df = G_df.groupby(list(G_df.columns.to_list())).size().reset_index(name=name)
        G_df = G_df.merge(G_selected_df, how='inner')
        
        return(G_df)

    def gof(self, G, selected, id_v):
        n, d = G.shape
        
        if d != 1:
            raise(ValueError("G is expected to have exactly 1 columns."))
        
        G_df = self.df_count(G, selected, columns=['g0'], name='E')
        G_df__v = self.df_count(G[id_v], selected[id_v], columns=['g0'], name='O')
        
        df = G_df.merge(G_df__v, how='outer')
        df['E'] = df['E'] / df['E'].sum() * df['O'].sum()
        
        df.fillna(0, inplace=True)
        
        chi2 = ((df['O'] - df['E'])**2 / df['E']).values.sum()
        
        return (chi2 / (df['E'].sum() * (df.index.size - 1)))**0.5
        

    def toi(self, G, selected, id_v, columns_id):
        n, d = G.shape
        
        if d != 2:
            raise(ValueError("G is expected to have exactly 2 columns."))
        G_df = self.df_count(G[id_v], selected[id_v], columns=['g' + str(k) for k in columns_id], name='O')
        # G_df__k0 = df_count(G[:,[0]][id_v], selected[id_v], columns=['g'+str(columns_id[0])], name='E_k'+str(columns_id[0]))
        # G_df__k1 = df_count(G[:,[1]][id_v], selected[id_v], columns=['g'+str(columns_id[1])], name='E_k'+str(columns_id[1]))
        
        # df = G_df.merge(G_df__k0, how='outer')
        # df = df.merge(G_df__k1, how='outer')
        
        N = G_df['O'].sum()
        # for k in columns_id:
            # df['E_k'+str(k)] = df['E_k'+str(k)] / df['E_k'+str(k)].sum() * N
        
        # df['E'] = df['E_k'+str(columns_id[0])] * df['E_k'+str(columns_id[1])] / N
        
        for k in columns_id:
            G_df['N'+str(k)] = G_df.groupby(['g'+str(k)])['O'].transform('sum')
        
        G_df['E'] = G_df['N'+str(columns_id[0])] * G_df['N'+str(columns_id[1])] / N
        
        # print(G_df)
        
        chi2 = ((G_df['O'] - G_df['E'])**2 / G_df['E']).values.sum()
        
        n_m = np.min([len(np.unique(G_df['g'+str(columns_id[0])].values)),
                       len(np.unique(G_df['g'+str(columns_id[1])].values))])
        
        if n_m - 1 <= 0:
            print('warning, this transition does not occur enough to be well calibrated.')
            return(1)
        
        return (chi2 / (N * (n_m - 1)))**0.5
    
    def mrmr_cramer(self, G, selected, id_v, ):
        n, d = G.shape
        V_gof = np.array([self.gof(G[:,[k1]], selected, id_v) for k1 in range(d)])
        
        evs = np.arange(d)[V_gof >= self.V_gof_min]
        print('V_gof', np.round(V_gof,4))
        print(V_gof.size)
        # for k1 in list(evs):
        
        evs = evs[np.argsort(V_gof[evs])[::-1]]
        
        list_k1_k2 = list(combinations(evs, 2))
        
        for k1, k2 in list_k1_k2:
            if k1 in evs and k2 in evs:
                V_toi = self.toi(G[:, [k1, k2]], selected, id_v, columns_id=[k1,k2])
                print(k1, k2, V_toi)
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
            selected = self.compute_selected(Z, id_v)
            ddx = self.ddx
            if ddx is None:
                ddx = self.compute_ddx(Z, selected, id_v)
        
            z_min = Z.min(axis=0)
            z_max = Z.max(axis=0)+0.0001 * Z.std(axis=0)
            dx = Z.std(axis=0)*ddx
            bins = [np.arange(z_min[k], z_max[k], dx[k]) for k in range(d)]
                    
            G = np.vstack([np.digitize(Z[:,k], bins[k]) for k in range(d)]).T
            self.G = G
            evs = self.mrmr_cramer(G, selected, id_v)
            
            print('v=',v, 'evs=', evs)
            
            id_evs[evs] = True
        
        evs = np.arange(d)[id_evs]
        print('==========')
        print('EVS : ', evs)
        return evs