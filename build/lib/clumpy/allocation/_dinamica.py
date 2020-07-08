from ._allocation import _Allocation
from ..calibration._calibration import _Calibration
from .. import definition
from ._patcher import _weighted_neighbors

import pandas as pd
import numpy as np
import copy
import time
from tqdm import tqdm

class Dinamica(_Allocation):
    """Dinamica Allocation Method
    
    Parameters
    ----------
    params : dict (default=None)
        the parameters dictionary
    """
    def __init__(self, params = None):
        super().__init__(params)
        
    def allocate_monopixel_patches(self,
                                   case:definition.Case,
                                   calibration=None,
                                   P_vf__vi=None,
                                   probability_maps=None,
                                   F=10,
                                   replace=True,
                                   k=500,
                                   sound=2,
                                   dict_args={}):
        """
        Fast allocation for monopixel patches

        Parameters
        ----------
        case : definition.Case
            Starting case which have to be discretized.
            
        calibration : _Calibration (default=None)
            Calibration object. If ``None``, the probability maps is expected.
            
        P_vf__vi : Pandas DataFrame (default=None)
            The transition matrix. If ``None``, the fitted ``self.P_vf__vi`` is used.
            
        probability_maps : definition.TransitionProbabilityLayers (default=None)
            The transition probabilities maps. If ``None``, it is computed according to the given case. It overwrites the calibration and ``P_vf__vi``.
            
        F : float (default=10)
            Pruning factor.
            
        replace : bool (default=True)
            Replace rejected pixels in the pool (source of allocation bias).
            
        k : float (default=1)
            factor of selected pixels.
        Returns
        -------
        map_f : definition.LandUseCoverLayer
            The allocated land use map
        """
        
        np.random.seed() # needed to seed in case of multiprocessing
        
        P_vf__vi=dict_args.get('P_vf__vi', P_vf__vi)
        probability_maps=dict_args.get('probability_maps', probability_maps)
        F=dict_args.get('F', F)
        replace=dict_args.get('replace', replace)
        k=dict_args.get('k', k)
        sound=dict_args.get('sound', sound)
        
        map_f_data = case.map_i.data.copy()
        
        if type(P_vf__vi) == type(None):
            P_vf__vi = calibration.P_vf__vi
        
        P_vf__vi = P_vf__vi.fillna(0)
        
        if type(probability_maps) == type(None):
            if type(P_vf__vi) == type(None):
                P_vf__vi = calibration.P_vf__vi
            
            probability_maps = calibration.transition_probability_maps(case, P_vf__vi)
        
        probability_maps = probability_maps.copy()
                
        map_f = definition.LandUseCoverLayer(name="luc_neutral",
                                   time=None,
                                   scale=case.map_i.scale,
                                   sound=0)
        map_f.import_numpy(data=map_f_data, sound=0)
            
        for vi in P_vf__vi.v.i.values.astype(int):
            
            J = case.discrete_J.loc[case.discrete_J.v.i==vi].copy()
                                
            # probability building
            for key, pm in probability_maps.layers.items():
                if key[0] == vi:
                    J[('P_vf__vi_z', key[1])] = 0
                    J.loc[J.v.i==key[0], [('P_vf__vi_z', key[1])]] = pm.data.flat[J.loc[J.v.i==key[0]].index.values]
            
            
            
            J.reset_index(drop=False, inplace=True)
            J.rename(columns={'index':'j'}, level=0, inplace=True)
                    
            # target volume computing
            N_vi_vf = {}
            for vf in J.P_vf__vi_z.columns.to_list():
                N_vi_vf[vf] = int(P_vf__vi.loc[P_vf__vi.v.i==vi].P_vf__vi[vf].values[0] * J.loc[J.v.i==vi].index.size)+1
            
            # pruning
            for vf in J.P_vf__vi_z.columns.to_list():
                J.sort_values(by=[('P_vf__vi_z',vf)], ascending=False, inplace=True)
                J.reset_index(drop=True, inplace=True)
                J[('trigger',vf)] = 0
                J.loc[J.index <= N_vi_vf[vf]*F, ('trigger',vf)] = 1
                   
            # keep only pixels with almost one possible transition
            J = J.loc[J.trigger.sum(axis=1) > 0]    
            J.drop(['trigger'], axis=1, level=0, inplace=True)
            
            J = J.loc[J.P_vf__vi_z.sum(axis=1) > 0]
            
            if not replace:
                J.P_vf__vi_z /= J.P_vf__vi_z.sum(axis=1).max()
            
            J['available'] = 1
            
            if sound > 0:
                print(J.index.size)
            
            while len(N_vi_vf) > 0:
                if J.loc[J.available==1].index.size == 0:
                    # on réarme ceux qui ont raté le premier test
                    if sound > 0:
                        print('not enought pixels. new draw')
                    J.loc[J.available==-1, 'available'] = 1
                
                if sound > 0:
                    print(J.loc[J.available==1].index.size, N_vi_vf)
                
                n = k*np.array(list(N_vi_vf.values())).sum()
                if n > J.loc[J.available==1].index.size:
                    n = J.loc[J.available==1].index.size
                
                j = J.loc[J.available==1].sample(n=n, replace=replace).copy()
                
                j.reset_index(drop=False, inplace=True)
                j.rename(columns={'index':'J_index'}, level=0, inplace=True)
                
                if replace:
                    j_selected = j.loc[np.random.random(j.index.size)<j.P_vf__vi_z.sum(axis=1)].copy()
                    j_selected.drop_duplicates(inplace=True)
                
                else:
                    j['x'] = np.random.random(j.index.size)
                    j_selected = j.loc[j.x<j.P_vf__vi_z.sum(axis=1)].copy()
                    
                    # return(j_selected)
                    
                    # on retire dans J les pixels non sélectionnés
                    J.loc[j.loc[j.x>=j.P_vf__vi_z.sum(axis=1)].J_index.values, 'available'] = 0
                    # J = J.loc[np.invert(J.index.isin(j.loc[j.x>=j.P_vf__vi_z.sum(axis=1)].J_index.values))].copy()
                    
                    j_selected.drop('x', axis=1, level=0, inplace=True)
                
                if sound > 0:
                    print('#j_selected', j_selected.index.size)
                    
                if j_selected.index.size > 0:                
                    
                    # on change P(vf|vi,z) pour être sûr que ça transite
                    j_selected.P_vf__vi_z /= j_selected.P_vf__vi_z.sum(axis=1).values[:,None]
                                
                    self._generalized_acceptation_rejection_test(j_selected, inplace=True, accepted_only=False)            
                    
                    # on applique le changement mono pixel
                    vf_list = list(N_vi_vf.keys())
                    for vf in vf_list:
                        j_selected_vf = j_selected.loc[j_selected.v.f == vf].copy()
                        j_selected_vf.reset_index(drop=True, inplace=True)
                        
                        if sound > 0:
                            print('vf', vf, '#j_selected_vf', j_selected_vf.index.size)
                        
                        # keep only enough pixels
                        j_selected_vf = j_selected_vf.loc[j_selected_vf.index < N_vi_vf[vf]].copy()
                        
                        # allocation
                        map_f_data.flat[j_selected_vf.j.values] = vf
                        N_vi_vf[vf] -= j_selected_vf.index.size
                        
                        J.loc[j_selected_vf.J_index.values, 'available'] = -1
                        
                        # J = J.loc[np.invert(J.index.isin(j_selected_vf.J_index.values))].copy()
                    
                        if N_vi_vf[vf] <= 0:
                            N_vi_vf.pop(vf)
                            J.drop(('P_vf__vi_z', vf), axis=1, inplace=True)
                            if len(N_vi_vf) > 0:
                                J = J.loc[J.P_vf__vi_z.sum(axis=1) > 0].copy()
                                
                                if not replace:
                                    J.P_vf__vi_z /= J.P_vf__vi_z.sum(axis=1).max()
                            if sound > 0:
                                print(str(vi)+'->'+str(vf)+' : done')
                
        map_f = definition.LandUseCoverLayer(name="luc_simple",
                                   time=None,
                                   scale=case.map_i.scale,
                                   sound=0)
        map_f.import_numpy(data=map_f_data, sound=0)
        
        return(map_f)
    
    def allocate(self,
                 case:definition.Case,
                 calibration=None,
                 P_vf__vi=None,
                 probability_maps=None,
                 F=10,
                 replace=True,
                 k=500,
                 sound=2,
                 dict_args={}):
        """
        Fast allocation like Dinamica. To reproduce exactly Dinamica behavior, use allocate_as_dinamica which is slower.

        Parameters
        ----------
        case : definition.Case
            Starting case which have to be discretized.
            
        calibration : _Calibration (default=None)
            Calibration object. If ``None``, the probability maps is expected.
            
        P_vf__vi : Pandas DataFrame (default=None)
            The transition matrix. If ``None``, the fitted ``self.P_vf__vi`` is used.
            
        probability_maps : definition.TransitionProbabilityLayers (default=None)
            The transition probabilities maps. If ``None``, it is computed according to the given case. It overwrites the calibration and ``P_vf__vi``.
            
        F : float (default=10)
            Pruning factor.
            
        replace : bool (default=True)
            Replace rejected pixels in the pool (source of allocation bias).
            
        k : float (default=1)
            factor of selected pixels.
            
        sound : int (default=2)
            Text output level. ``0`` means silent mode.
            
        dict_args : dict (default=``{}``)
            The above optional arguments in a dictionary. Overwrites if already passed. 
        Returns
        -------
        map_f : definition.LandUseCoverLayer
            The allocated land use map
        """
        
        np.random.seed() # needed to seed in case of multiprocessing
        
        P_vf__vi=dict_args.get('P_vf__vi', P_vf__vi)
        probability_maps=dict_args.get('probability_maps', probability_maps)
        F=dict_args.get('F', F)
        replace=dict_args.get('replace', replace)
        k=dict_args.get('k', k)
        sound=dict_args.get('sound', sound)
        
        map_f_data = case.map_i.data.copy()
        
        if type(P_vf__vi) == type(None):
            P_vf__vi = calibration.P_vf__vi
        
        if type(probability_maps) == type(None):
            if type(P_vf__vi) == type(None):
                P_vf__vi = calibration.P_vf__vi
            
            probability_maps = calibration.transition_probability_maps(case, P_vf__vi)
        
        probability_maps = probability_maps.copy()
        
        
        map_f = definition.LandUseCoverLayer(name="luc_neutral",
                                   time=None,
                                   scale=case.map_i.scale)
        map_f.import_numpy(data=map_f_data, sound=sound)
                
        for vi in P_vf__vi.v.i.values.astype(int):
            
            J = create_J(case.map_i)
            
            J = J.loc[J.v.i==vi]
            
    
                    
            # probability building
            for key, pm in probability_maps.layers.items():
                if key[0] == vi:
                    J[('P_vf__vi_z', key[1])] = 0
                    J.loc[J.v.i==key[0], [('P_vf__vi_z', key[1])]] = pm.data.flat[J.loc[J.v.i==key[0]].index.values]
                    
    
            J.reset_index(drop=False, inplace=True)
            J.rename(columns={'index':'j'}, level=0, inplace=True)
                    
            # target volume computing
            N_vi_vf = {}
            for vf in P_vf__vi.P_vf__vi.columns.to_list():
                N_vi_vf[vf] = int(P_vf__vi.loc[P_vf__vi.v.i==vi].P_vf__vi[vf].values[0] * J.loc[J.v.i==vi].index.size)+1
            
            
            
            # pruning
            for vf in P_vf__vi.P_vf__vi.columns.to_list():
                J.sort_values(by=[('P_vf__vi_z',vf)], ascending=False, inplace=True)
                J.reset_index(drop=True, inplace=True)
                J[('trigger',vf)] = 0
                J.loc[J.index <= N_vi_vf[vf]*F, ('trigger',vf)] = 1
                   
            # keep only pixels with almost one possible transition
            J = J.loc[J.trigger.sum(axis=1) > 0]    
            J.drop(['trigger'], axis=1, level=0, inplace=True)
            
            J = J.loc[J.P_vf__vi_z.sum(axis=1) > 0]
            
            if not replace:
                J.P_vf__vi_z /= J.P_vf__vi_z.sum(axis=1).max()
            
            J['available'] = 1
            
            if sound > 1:
                print(J.index.size)
            
            while len(N_vi_vf) > 0:
                if J.loc[J.available==1].index.size == 0:
                    # on réarme ceux qui ont raté le premier test
                    if sound > 1:
                        print('not enought pixels. new draw')
                    J.loc[J.available==-1, 'available'] = 1
                if sound > 1:
                    print(J.loc[J.available==1].index.size, N_vi_vf)
                
                n = k*np.array(list(N_vi_vf.values())).sum()
                if n > J.loc[J.available==1].index.size:
                    n = J.loc[J.available==1].index.size
                
                j = J.loc[J.available==1].sample(n=n, replace=replace).copy()
                
                j.reset_index(drop=False, inplace=True)
                j.rename(columns={'index':'J_index'}, level=0, inplace=True)
                
                if replace:
                    j_selected = j.loc[np.random.random(j.index.size)<j.P_vf__vi_z.sum(axis=1)].copy()
                    j_selected.drop_duplicates(inplace=True)
                
                else:
                    j['x'] = np.random.random(j.index.size)
                    j_selected = j.loc[j.x<j.P_vf__vi_z.sum(axis=1)].copy()
                    
                    # return(j_selected)
                    
                    # on retire dans J les pixels non sélectionnés
                    J.loc[j.loc[j.x>=j.P_vf__vi_z.sum(axis=1)].J_index.values, 'available'] = 0
                    # J = J.loc[np.invert(J.index.isin(j.loc[j.x>=j.P_vf__vi_z.sum(axis=1)].J_index.values))].copy()
                    
                    j_selected.drop('x', axis=1, level=0, inplace=True)
                if sound > 1:    
                    print('#j_selected', j_selected.index.size)
                    
                if j_selected.index.size > 0:                
                    
                    # on change P(vf|vi,z) pour être sûr que ça transite
                    j_selected.P_vf__vi_z /= j_selected.P_vf__vi_z.sum(axis=1).values[:,None]
                                
                    self._generalized_acceptation_rejection_test(j_selected, inplace=True, accepted_only=False)            
                    
                    list_vi = [(vi, vf) for vf in N_vi_vf.keys()]
                    self._draw_patches_parameters(j_selected, list_vi)
                    
                    keep_allocate = True
                    while keep_allocate and j_selected.index.size > 0:
                        row_j = j_selected.sample(n=1)
                        
                        j = row_j.j.values[0].astype(int)
                        vi = row_j.v.i.values[0]
                        vf = row_j.v.f.values[0]
                        S_patch = row_j.S_patch.values[0]
                        if type(self.params[(vi,vf)]['patches_parameters']['isl']['eccentricity']) == type(None):
                            eccentricity_mean = None
                            eccentricity_std = None
                        else:
                            eccentricity_mean = self.params[(vi,vf)]['patches_parameters']['isl']['eccentricity']['mean']
                            eccentricity_std = self.params[(vi,vf)]['patches_parameters']['isl']['eccentricity']['std']
                        neighbors_structure = self.params[(vi,vf)]['patches_parameters']['isl']['neighbors_structure']
                        avoid_aggregation = self.params[(vi,vf)]['patches_parameters']['isl']['avoid_aggregation']
                        nb_of_neighbors_to_fill = self.params[(vi,vf)]['patches_parameters']['isl']['nb_of_neighbors_to_fill']
                        proceed_even_if_no_probability = self.params[(vi,vf)]['patches_parameters']['isl']['proceed_even_if_no_probability']
                        
                        # patch design around the kernel pixel
                        S = _weighted_neighbors(map_i_data = case.map_i.data,
                                                     map_f_data=map_f_data,
                                                     map_P_vf__vi_z=probability_maps.layers[(vi,vf)].data,
                                                     j_kernel=j,
                                                     vi=vi,
                                                     vf=vf,
                                                     patch_S=S_patch,
                                                     eccentricity_mean=eccentricity_mean,
                                                     eccentricity_std=eccentricity_std,
                                                     neighbors_structure=neighbors_structure,
                                                     avoid_aggregation=avoid_aggregation,
                                                     nb_of_neighbors_to_fill=nb_of_neighbors_to_fill,
                                                     proceed_even_if_no_probability=proceed_even_if_no_probability)
                        
                        N_vi_vf[vf] -= S
                        
                        j_selected.drop(row_j.index.values[0], axis=0, inplace=True)
                        
                        if N_vi_vf[vf] <= 0:
                            N_vi_vf.pop(vf)
                            if len(N_vi_vf) > 0:
                                # on enleve les probas de cette transition
                                J.loc[J.v.i==vi, ('P_vf__vi_z',vf)] = 0
                                J = J.loc[J.P_vf__vi_z.sum(axis=1) > 0].copy()
                            keep_allocate = False
                            if not replace:
                                    J.P_vf__vi_z /= J.P_vf__vi_z.sum(axis=1).max()
                            if sound > 1:
                                print(str(vi)+'->'+str(vf)+' : done')
                                    
        map_f = definition.LandUseCoverLayer(name="luc_simple",
                                   time=None,
                                   scale=case.map_i.scale,
                                   sound=sound)
        map_f.import_numpy(data=map_f_data, sound=sound)
        
        return(map_f)
    
    def allocate_as_dinamica(self, calibration:_Calibration, case:definition.Case, P_vf__vi=None, F=10, replace=True):
        """
        Allocation as Dinamica. To have an equivalent faster process, please use allocate.

        Parameters
        ----------
        calibration : _Calibration
            Calibration object.
        case : definition.Case
            Starting case which have to be discretized.
        P_vf__vi : Pandas DataFrame (default=None)
            The transition matrix. If ``None``, the fitted ``self.P_vf__vi`` is used.
        F : float (default=10)
            Pruning factor.
        replace : bool (default=True)
            Replace rejected pixels in the pool (source of allocation bias).
        Returns
        -------
        map_f : definition.LandUseCoverLayer
            The allocated land use map
        """
        
        map_f_data = case.map_i.data.copy()
        
        
        if type(P_vf__vi) == type(None):
            P_vf__vi = calibration.P_vf__vi
        
        probability_maps = calibration.transition_probability_maps(case, P_vf__vi)
        
        map_f = definition.LandUseCoverLayer(name="luc_neutral",
                                   time=None,
                                   scale=case.map_i.scale)
        map_f.import_numpy(data=map_f_data)
            
        for vi in P_vf__vi.v.i.values.astype(int):
            
            J = create_J(case.map_i)
            
            J = J.loc[J.v.i==vi]
        
            # probability building
            for key, pm in probability_maps.layers.items():
                if key[0] == vi:
                    J[('P_vf__vi_z', key[1])] = 0
                    J.loc[J.v.i==key[0], [('P_vf__vi_z', key[1])]] = pm.data.flat[J.loc[J.v.i==key[0]].index.values]
            
            
            
            J.reset_index(drop=False, inplace=True)
            J.rename(columns={'index':'j'}, level=0, inplace=True)
                    
            # target volume computing
            N_vi_vf = {}
            for vf in P_vf__vi.P_vf__vi.columns.to_list():
                N_vi_vf[vf] = int(P_vf__vi.loc[P_vf__vi.v.i==vi].P_vf__vi[vf].values[0] * J.loc[J.v.i==vi].index.size)+1
                
            # pruning
            for vf in P_vf__vi.P_vf__vi.columns.to_list():
                J.sort_values(by=[('P_vf__vi_z',vf)], ascending=False, inplace=True)
                J.reset_index(drop=True, inplace=True)
                J[('trigger',vf)] = 0
                J.loc[J.index <= N_vi_vf[vf]*F, ('trigger',vf)] = 1
                   
            # keep only pixels with almost one possible transition
            J = J.loc[J.trigger.sum(axis=1) > 0]    
            J.drop(['trigger'], axis=1, level=0, inplace=True)
            
            J[('S','')] = J.P_vf__vi_z.sum(axis=1)
            J = J.loc[J.S > 0]
            
            print(J.index.size)
    
            # allocation
            pbar = tqdm(total=np.array(list(N_vi_vf.values())).sum())
            up = 0
            while len(N_vi_vf) > 0:
                if J.index.size>0:
                    j = J.sample(n=1)
                                
                    if np.random.random() < j.S.values[0]:
                        
                        try_alloc = True                                                
                        while try_alloc:
                            
                            # vf = np.random.choice(list_vf)
                            vf = np.random.choice(list(N_vi_vf.keys()))
                            
                            if np.random.random() < j.P_vf__vi_z[vf].values[0] / j.S.values[0]:
                                try_alloc = False
                                map_f_data.flat[j.j.values[0]] = vf
                                N_vi_vf[vf] -= 1
                                J.drop(j.index, axis=0, inplace=True)
                        
                                if N_vi_vf[vf] <= 0:
                                    N_vi_vf.pop(vf)
                                    J.drop(('P_vf__vi_z', vf), axis=1, inplace=True)
                                    if len(N_vi_vf) > 0:
                                        J[('S','')] = J.P_vf__vi_z.sum(axis=1)
                                        J = J.loc[J.S > 0]
                                        print(J.index.size)
                                    print(str(vi)+'->'+str(vf)+' : done')
                                
                                up += 1
                                if up % 100==0:
                                    pbar.update(100)
                    elif not replace:
                        J.drop(j.index, axis=0, inplace=True)
                else:
                    break
                
        map_f = definition.LandUseCoverLayer(name="luc_simple",
                                   time=None,
                                   scale=case.map_i.scale)
        map_f.import_numpy(data=map_f_data)
            
        
        return(map_f)
    
def create_J(layer_LUC_i, layer_LUC_f=None):    
    # all pixels with vi value
    cols = [('v', 'i')]
    cols = pd.MultiIndex.from_tuples(cols)
    
    J = pd.DataFrame(layer_LUC_i.data.flat, columns=cols)
            
    return(J)