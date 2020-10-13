from ._allocation import _Allocation, compute_P_vf__vi_from_transition_probability_maps
# from ..calibration._calibration import _Calibration
from .. import definition
from ._patcher import _weighted_neighbors
from .scenario import compute_P_vf__vi_from_transition_probabilities
from ..tools import draw_within_histogram
from ..calibration import compute_P_z__vi
from tqdm import tqdm

import numpy as np
import pandas as pd
import time
from copy import deepcopy

class SimpleUnbiased(_Allocation):
    """Simple Unbiased Allocation Method
    
    Parameters
    ----------
    params : dict (default=None)
        the parameters dictionary
    """
    
    def __init__(self, params = None):
        super().__init__(params)
    
    def _allocate_monopixel_patches(self,
                                   case,
                                   tp,
                                   sound=2):
        """
        Simple allocation of monopixels patches whithout scenario control.

        Parameters
        ----------
            
                    
        sound : int (default=2)
            Text output level. ``0`` means silent mode.
            
        dict_args : dict (default=``{}``)
            The above optional arguments in a dictionary. Overwrites if already passed. 

        Returns
        -------
        map_f : definition.LandUseCoverLayer
            The allocated land use map
            
        Notes
        -----
        New attributes are availables :`
            
            ``self.execution_time`` One microseconds precision for Linux and Mac OS and 16 milliseconds precision for Windows.
        
            ``self.detailed_execution_time`` One microseconds precision for Linux and Mac OS and 16 milliseconds precision for Windows.
            
            ``self.tested_pixels``
        """
        np.random.seed() # needed to seed in case of multiprocessing
        
        global_start_time = time.time()
        start_time = time.time()
                    
        map_f_data = case.map_i.data.copy()
        
        self.detailed_execution_time = {}
        
        self.detailed_execution_time['initialization']=time.time()-start_time
        start_time = time.time()
                
        # GART
        self.tested_pixels = [np.sum([tp_vi.shape[0] for tp_vi in tp.values()])]
        vf = self._generalized_acceptation_rejection_test(tp, case.dict_vi_vf)
        
        print((vf[3] == 7).mean())
        print((vf[3] == 8).mean())
        
        self.detailed_execution_time['sampling']=[time.time()-start_time]
        start_time = time.time()
                
        # allocation
        for vi in tp.keys():
            map_f_data.flat[case.J[vi]] = vf[vi]
        
        self.detailed_execution_time['pixels_initialization']=time.time()-start_time

        self.detailed_execution_time['patches_parameters_initialization']=[0]
        
        self.execution_time = time.time() - global_start_time
        
        # post processing
        map_f = definition.LandUseCoverLayer(name="luc_simple",
                                               time=None,
                                               scale=case.map_i.scale)
        map_f.import_numpy(data=map_f_data, sound=sound)
        
        if sound>0:
            print('FINISHED')
            print('========')
            print('execution times')
            print(self.detailed_execution_time)
            
            
            
            # N_vi_vf = J.groupby([('v','i'), ('v','f')]).size().reset_index(name=('N_vi_vf', ''))
            # N_vi = J_proba.groupby([('v','i')]).size().reset_index(name=('N_vi', ''))
            # N_vi_vf = N_vi_vf.merge(N_vi)
            # N_vi_vf['P_vf__vi'] = N_vi_vf.N_vi_vf / N_vi_vf.N_vi
            # print(N_vi_vf)
        
        return(map_f)
    
    def allocate(self,
                  case,
                  tp,
                  isl_exp_ratio,
                  h_area,
                  h_eccentricity,
                  neighbors_structure='rook',
                  avoid_aggregation=False,
                  nb_of_neighbors_to_fill=3,
                  proceed_even_if_no_probability=True,
                  sound=2):
        
        np.random.seed() # needed to seed in case of multiprocessing
        
        global_start_time = time.time()
        start_time = time.time()
                    
        map_f_data = case.map_i.data.copy()
        
        P_z__vi = compute_P_z__vi(case)
                   
                
        self.detailed_execution_time = {}
        
        self.detailed_execution_time['initialization']=time.time()-start_time
        start_time = time.time()
        
        
        for vi in tp.keys():
            P_z__vi_all_J_vi = case.get_z_as_dataframe()[vi].merge(right=P_z__vi[vi],
                                            how='left')
            for isl_exp in ['isl', 'exp']:
                r = []
                averaged_area = []
                for vf in case.dict_vi_vf[vi]:
                     # first isl_exp edit
                    r.append(isl_exp_ratio[vi][vf])
                    if isl_exp == 'exp':
                        r[-1] = 1-r[-1]
                    
                    # surface coef
                    a = (h_area[vi][vf][isl_exp][1][:-1]+np.diff(h_area[vi][vf][isl_exp][1])/2)
                    w = h_area[vi][vf][isl_exp][0]
                    averaged_area.append(np.average(a, weights=w))    
                
                r = np.array(r)
                averaged_area = np.array(averaged_area)
                
                tp_vi = tp[vi].copy()
                
                # if expansion, exclude too far pixels
                if isl_exp == 'exp':
                    c_all_vf = np.zeros(case.J[vi].shape).astype(bool)
                    for id_vf, vf in enumerate(case.dict_vi_vf[vi]):
                        c = case.get_z(vi=vi, z_name='distance_to_'+str(vf)) == 1
                        tp_vi[c, id_vf] = 0
                        
                        c_all_vf = c_all_vf | c
                    
                    case_exp = case.keep_only(vi=vi, condition=c_all_vf)
                    
                    # P_z__vi has to be edited
                    P_z__vi_case_exp = compute_P_z__vi(case_exp, name='P_z__vi_case_exp')
                    P_z__vi_case_exp_all_J = P_z__vi_all_J_vi.merge(right=P_z__vi_case_exp[vi],
                                                           how='left')
                    
                    P_z__vi_case_exp_all_J.fillna(0, inplace=True)
                    
                    tp_vi = tp_vi * (P_z__vi_case_exp_all_J.P_z__vi_case_exp.values / P_z__vi_case_exp_all_J.P_z__vi.values)[:, None]
                    
                    
                # gart
                tp_vi = tp_vi*r/averaged_area
                print(averaged_area)
                if tp_vi.sum(axis=1).max() > 1:
                    print('Warning ! probability > 1')
                    
                gart = self._generalized_acceptation_rejection_test_vi(vi,
                                                                       tp_vi,
                                                                       case.dict_vi_vf[vi])
                
                # if expansion, tp should be not null for the patcher
                if isl_exp == 'exp':
                    tp_vi = tp[vi].copy()
                
                id_J_to_keep = gart != vi
                
                J = case.J[vi][id_J_to_keep]
                vf_J = gart[id_J_to_keep]
                                
                area = np.zeros(J.shape)
                eccentricity = np.zeros(J.shape)
                map_tp = {}
                for id_vf, vf in enumerate(case.dict_vi_vf[vi]):
                    idx = vf_J == vf
                    area[idx] = draw_within_histogram(bins = h_area[vi][vf][isl_exp][1],
                                                    p = h_area[vi][vf][isl_exp][0],
                                                    n = idx.sum())
                
                    eccentricity[idx] = draw_within_histogram(bins = h_eccentricity[vi][vf][isl_exp][1],
                                                    p = h_eccentricity[vi][vf][isl_exp][0],
                                                    n = idx.sum())
                    
                    map_tp[vf] = np.zeros(map_f_data.shape)
                    map_tp[vf].flat[J] = tp_vi[id_J_to_keep, id_vf]
                    
                    print(vf, area[idx].mean())
                    # print(vf, area[idx].sum()/case.J[vi].size)
                    
                    
                id_J_sample = np.random.choice(np.arange(J.size),
                                               size=J.size,
                                               replace=False)
                
                # for id_J in tqdm(id_J_sample):
                #     _weighted_neighbors(map_i_data = case.map_i.data,
                #                   map_f_data=map_f_data,
                #                   map_P_vf__vi_z=map_tp[vf_J[id_J]],
                #                   j_kernel=J[id_J],
                #                   vi=vi,
                #                   vf=vf_J[id_J],
                #                   patch_S=area[id_J],
                #                   eccentricity_mean=eccentricity[id_J],
                #                   eccentricity_std=0.05,
                #                   neighbors_structure=neighbors_structure,
                #                   avoid_aggregation=avoid_aggregation,
                #                   nb_of_neighbors_to_fill=nb_of_neighbors_to_fill,
                #                   proceed_even_if_no_probability=proceed_even_if_no_probability)
                    
                
        
        self.execution_time = time.time() - global_start_time
        
        # post processing
        map_f = definition.LandUseCoverLayer(name="luc_simple",
                                               time=None,
                                               scale=case.map_i.scale)
        map_f.import_numpy(data=map_f_data, sound=sound)
        
        if sound>0:
            print('FINISHED')
            print('========')
            print('execution times')
            print(self.detailed_execution_time)
            print('total', round(self.execution_time,2), 's')
            
        return(map_f)
        
    def _allocate2(self,
                 case:definition.Case,
                 probability_maps=None,
                 update='none',
                 sound=2,
                 dict_args={}):
        """
        Parameters
        ----------
        case : definition.Case
            Starting case which have to be discretized.
            
        probability_maps : definition.TransitionProbabilityLayers (default=None)
            The transition probabilities maps. If ``None``, it is computed according to the given case. It overwrites the calibration and ``P_vf__vi``.
        
        update : {'none', 'transition', 'ghost', 'both'}, (default='none')
            The P(z|vi,vf) update policy.
            
            none
                no update
                
            transition
                only when a transition is achieved.
                
            ghost
                only when the ghost tolerance is reached.
                
            both
                for both transition and ghost modes.
                
        sound : int (default=2)
            Text output level. ``0`` means silent mode.
            
        dict_args : dict (default=``{}``)
            The above optional arguments in a dictionary. Overwrites if already passed. 
        
        Returns
        -------
        map_f : definition.LandUseCoverLayer
            The allocated land use map

        Notes
        -----
        New attributes are availables :
            
            ``self.ghost_allocation``
            
            ``self.execution_time`` One microseconds precision for Linux and Mac OS and 16 milliseconds precision for Windows.
            
            ``self.detailed_execution_time`` One microseconds precision for Linux and Mac OS and 16 milliseconds precision for Windows.
            
            ``self.tested_pixels``
            
            ``self.infos``
            
            ``self.N_vi_vf``
            
            ``self.N_vi_vf_target``
        """
        
        np.random.seed() # needed to seed in case of multiprocessing
        
        probability_maps=dict_args.get('probability_maps', probability_maps)
        update=dict_args.get('update', update)
        sound=dict_args.get('sound', sound)
        
        try:
            probability_maps = probability_maps.copy()
        except:
            raise TypeError('unexpected probability_maps')
        
        self.detailed_execution_time = {}
        global_start_time = time.time()
        start_time = time.time()
        P_vf__vi = compute_P_vf__vi_from_transition_probability_maps(case, probability_maps)
        
        # print(P_vf__vi)
        
        self.detailed_execution_time['transition_matrix']=time.time()-start_time
        start_time = time.time()
        
        J = case.discrete_J.copy()
        
        map_f_data = case.map_i.data.copy()
        
        N_vi_vf = self._compute_N_vi_vf(list(probability_maps.layers.keys()), J, P_vf__vi)
        N_vi_vf_target = N_vi_vf.copy()
        if sound > 0:
            print(N_vi_vf)
            
        # build the P_z__vi map 
        calibration = _Calibration()
        calibration.compute_P_z__vi(case)
        M_P_z__vi = calibration.build_P_z__vi_map(case)
        
        self.detailed_execution_time['pixels_initialization']=time.time()-start_time
        start_time = time.time()
        
        # select pivot-cells
        self.detailed_execution_time['sampling'] = []
        self.detailed_execution_time['patches_parameters_initialization'] = []
        
        self.tested_pixels = []
        
        J_pivotcells = self._sample(J=J,
                                probability_maps=probability_maps)
        
        if type(J_pivotcells) == type(False):
            if sound > 0:
                print('Error during sampling')
            return(False)
        
        if J_pivotcells.index.size == 0 and sound > 0:
            print('The sampling returned no pixels. Tested pixels: '+str(self.tested_pixels[-1]))
                
        idx = -1
        ghost_allocation = dict.fromkeys(N_vi_vf.keys(), 0)
        total_ghost_allocation = dict.fromkeys(N_vi_vf.keys(), 0)
        while J_pivotcells.index.size > 0:
            # draw one kernel pixel
            idx += 1
            
            j = J_pivotcells.loc[idx,'j'].values[0].astype(int)
            vi = J_pivotcells.loc[idx,('v','i')]
            vf = J_pivotcells.loc[idx,('v','f')]
            S_patch = J_pivotcells.loc[idx,('S_patch')].values[0]
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
            
            if S == 0:
                ghost_allocation[(vi,vf)] += S_patch
                total_ghost_allocation[(vi,vf)] += S_patch
            
            # update N_vi_vf
            N_vi_vf[(vi,vf)] -= S
            
            update_P_z__vi_trigger = False
            
            if N_vi_vf[(vi,vf)] <= 0:
                if sound > 0:
                    print('The transition ('+str(vi)+','+str(vf)+') is achieved.')
                if update in ['transition', 'both']:
                    update_P_z__vi_trigger = True
                else:
                    # remove concerned pixels
                    J_pivotcells.drop(J_pivotcells.loc[(J_pivotcells.v.i==vi) &
                                                       (J_pivotcells.v.f==vf)].index.values,
                                      axis=0,
                                      inplace=True)
                    J_pivotcells.reset_index(drop=True, inplace=True)
                    idx = 0
                
            if update in ['ghost', 'both']:
                if ghost_allocation[(vi,vf)] > N_vi_vf_target[(vi,vf)]*self.params[(vi,vf)]['patches_parameters']['isl']['ghost_tolerance']:
                    if sound > 0:
                        print('ghost threshold reached for ('+str(vi)+','+str(vf)+')')
                    update_P_z__vi_trigger = True
                    ghost_allocation[(vi,vf)] = 0
            
            # on met à jour les probabilités
            if update_P_z__vi_trigger:
                if sound > 0:
                    print('mise à jour')
                    print(N_vi_vf)
                # on met à jour J_initial en retirant les pixels qui ont transités
                J[('v','f')] = map_f_data.flat[J.index.values]
                J.drop(J.loc[J.v.i != J.v.f].index.values, axis=0, inplace=True)
                # on calcule le nouveau P_z__vi (bien indiquer le nouveau J)
                calibration.compute_P_z__vi(case, J=J)
                # on peut donc calculer la nouvelle map P_z__vi
                M_P_z__vi_new = calibration.build_P_z__vi_map(case, J=J)
                # on met à jour les cartes de proba
                for key, map_layer in probability_maps.layers.items():
                    if N_vi_vf[key] > 0:  # si l'objectif n'est pas atteint
                        map_layer.data *=  np.divide(M_P_z__vi, M_P_z__vi_new, out=np.zeros(map_f_data.shape), where=M_P_z__vi_new!=0)
                    else: # sinon, on met à 0 cette transition
                        map_layer.data.fill(0)
                # la nouvelle carte prend la place de l'actuelle
                M_P_z__vi = M_P_z__vi_new
                # calcul des nouveaux pivot-cells
                J_pivotcells = self._sample(J=J, probability_maps=probability_maps)
                # le compteur idx est réinitialisé
                idx = -1
        
        self.detailed_execution_time['allocation']=time.time()-start_time - np.sum(self.detailed_execution_time['sampling']) - np.sum(self.detailed_execution_time['patches_parameters_initialization'])
        
        self.execution_time = time.time()-global_start_time
        
        
                
        map_f = definition.LandUseCoverLayer(name="luc_simple",
                                   time=None,
                                   scale=case.map_i.scale,
                                   sound=sound)
        map_f.import_numpy(data=map_f_data, sound=sound)
        
        if sound > 0:
            print('FINISHED')
            print('========')
        
        # some informations are saved
        self.N_vi_vf = N_vi_vf
        self.N_vi_vf_target = N_vi_vf_target
        self.ghost_allocation = total_ghost_allocation
        
        infos = pd.DataFrame(columns=['vi','vf','N_vi_vf', 'N_vi_vf_target', 'ratio', 'ghost', 'ghost_ratio'])
        
        for vi_vf in N_vi_vf.keys():
            if N_vi_vf_target[vi_vf]-N_vi_vf[vi_vf] > 0:
                ghost_ratio = total_ghost_allocation[vi_vf]/(N_vi_vf_target[vi_vf]-N_vi_vf[vi_vf])
            else:
                ghost_ratio = np.nan
            infos.loc[infos.index.size] = [vi_vf[0],
                                           vi_vf[1],
                                           N_vi_vf_target[vi_vf]-N_vi_vf[vi_vf],
                                           N_vi_vf_target[vi_vf],
                                           (N_vi_vf_target[vi_vf]-N_vi_vf[vi_vf])/N_vi_vf_target[vi_vf],
                                           total_ghost_allocation[vi_vf],
                                           ghost_ratio]
            
        if sound > 0:
            print(infos)
        
        self.infos = infos
        
        if sound > 0:
            print('execution times')
            print(self.detailed_execution_time)
        
        return(map_f)
    
    def allocate_with_expansion(self,
                                case:definition.Case,
                                probability_maps=None,
                                update='none',
                                sound=2,
                                dict_args={}):
        """
        Parameters
        ----------
        case : definition.Case
            Starting case which have to be discretized.
            
        probability_maps : definition.TransitionProbabilityLayers (default=None)
            The transition probabilities maps. If ``None``, it is computed according to the given case. It overwrites the calibration and ``P_vf__vi``.
        
        update : {'none', 'transition', 'ghost', 'both'}, (default='none')
            The P(z|vi,vf) update policy.
            
            none
                no update
                
            transition
                only when a transition is achieved.
                
            ghost
                only when the ghost tolerance is reached.
                
            both
                for both transition and ghost modes.
                
        sound : int (default=2)
            Text output level. ``0`` means silent mode.
            
        dict_args : dict (default=``{}``)
            The above optional arguments in a dictionary. Overwrites if already passed. 
        
        Returns
        -------
        map_f : definition.LandUseCoverLayer
            The allocated land use map

        Notes
        -----
        New attributes are availables :
            
            ``self.ghost_allocation``
            
            ``self.detailed_execution_time`` One microseconds precision for Linux and Mac OS and 16 milliseconds precision for Windows.
            
            ``self.tested_pixels``
            
            ``self.infos``
            
            ``self.N_vi_vf``
            
            ``self.N_vi_vf_target``
        """
        
        np.random.seed() # needed to seed in case of multiprocessing
        
        probability_maps=dict_args.get('probability_maps', probability_maps)
        update=dict_args.get('update', update)
        sound=dict_args.get('sound', sound)
        
        try:
            probability_maps = probability_maps.copy()
        except:
            raise TypeError('unexpected probability_maps')
        
        self.detailed_execution_time = {}
                
        start_time = time.time()
        P_vf__vi = compute_P_vf__vi_from_transition_probability_maps(case, probability_maps)
        
        self.detailed_execution_time['transition_matrix']=time.time()-start_time
        start_time = time.time()
        
        J = case.discrete_J.copy()
        
        map_f_data = case.map_i.data.copy()
        
        N_vi_vf = self._compute_N_vi_vf(list(probability_maps.layers.keys()), J, P_vf__vi)
        N_vi_vf_target = N_vi_vf.copy()
        if sound > 0:
            print(N_vi_vf)
            
        # build the P_z__vi map 
        calibration = _Calibration()
        calibration.compute_P_z__vi(case)
        M_P_z__vi = calibration.build_P_z__vi_map(case)
        
        self.detailed_execution_time['pixels_initialization']=time.time()-start_time
        start_time = time.time()
        
        # select pivot-cells
        self.detailed_execution_time['sampling'] = []
        self.detailed_execution_time['patches_parameters_initialization'] = []
        
        self.tested_pixels = []
        
        J_pivotcells = self._sample(J=J,
                                probability_maps=probability_maps)
        
        if type(J_pivotcells) == type(False):
            if sound > 0:
                print('Error during sampling')
            return(False)
        
        if J_pivotcells.index.size == 0 and sound > 0:
            print('The sampling returned no pixels. Tested pixels: '+str(self.tested_pixels[-1]))
                
        idx = -1
        ghost_allocation = dict.fromkeys(N_vi_vf.keys(), 0)
        total_ghost_allocation = dict.fromkeys(N_vi_vf.keys(), 0)
        while J_pivotcells.index.size > 0:
            # draw one kernel pixel
            idx += 1
            
            j = J_pivotcells.loc[idx,'j'].values[0].astype(int)
            vi = J_pivotcells.loc[idx,('v','i')]
            vf = J_pivotcells.loc[idx,('v','f')]
            S_patch = J_pivotcells.loc[idx,('S_patch')].values[0]
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
            
            if S == 0:
                ghost_allocation[(vi,vf)] += S_patch
                total_ghost_allocation[(vi,vf)] += S_patch
            
            # update N_vi_vf
            N_vi_vf[(vi,vf)] -= S
            
            update_P_z__vi_trigger = False
            
            if N_vi_vf[(vi,vf)] <= 0:
                if sound > 0:
                    print('The transition ('+str(vi)+','+str(vf)+') is achieved.')
                if update in ['transition', 'both']:
                    update_P_z__vi_trigger = True
                else:
                    # remove concerned pixels
                    J_pivotcells.drop(J_pivotcells.loc[(J_pivotcells.v.i==vi) &
                                                       (J_pivotcells.v.f==vf)].index.values,
                                      axis=0,
                                      inplace=True)
                    J_pivotcells.reset_index(drop=True, inplace=True)
                    idx = 0
                
            if update in ['ghost', 'both']:
                if ghost_allocation[(vi,vf)] > N_vi_vf_target[(vi,vf)]*self.params[(vi,vf)]['patches_parameters']['isl']['ghost_tolerance']:
                    if sound > 0:
                        print('ghost threshold reached for ('+str(vi)+','+str(vf)+')')
                    update_P_z__vi_trigger = True
                    ghost_allocation[(vi,vf)] = 0
            
            # on met à jour les probabilités
            if update_P_z__vi_trigger:
                if sound > 0:
                    print('mise à jour')
                    print(N_vi_vf)
                # on met à jour J_initial en retirant les pixels qui ont transités
                J[('v','f')] = map_f_data.flat[J.index.values]
                J.drop(J.loc[J.v.i != J.v.f].index.values, axis=0, inplace=True)
                # on calcule le nouveau P_z__vi (bien indiquer le nouveau J)
                calibration.compute_P_z__vi(case, J=J)
                # on peut donc calculer la nouvelle map P_z__vi
                M_P_z__vi_new = calibration.build_P_z__vi_map(case, J=J)
                # on met à jour les cartes de proba
                for key, map_layer in probability_maps.layers.items():
                    if N_vi_vf[key] > 0:  # si l'objectif n'est pas atteint
                        map_layer.data *=  np.divide(M_P_z__vi, M_P_z__vi_new, out=np.zeros(map_f_data.shape), where=M_P_z__vi_new!=0)
                    else: # sinon, on met à 0 cette transition
                        map_layer.data.fill(0)
                # la nouvelle carte prend la place de l'actuelle
                M_P_z__vi = M_P_z__vi_new
                # calcul des nouveaux pivot-cells
                J_pivotcells = self._sample(J=J, probability_maps=probability_maps)
                # le compteur idx est réinitialisé
                idx = -1
        
        self.detailed_execution_time['allocation']=time.time()-start_time - np.sum(self.detailed_execution_time['sampling']) - np.sum(self.detailed_execution_time['patches_parameters_initialization'])
                
        map_f = definition.LandUseCoverLayer(name="luc_simple",
                                   time=None,
                                   scale=case.map_i.scale,
                                   sound=sound)
        map_f.import_numpy(data=map_f_data, sound=sound)
        
        if sound > 0:
            print('FINISHED')
            print('========')
        
        # some informations are saved
        self.N_vi_vf = N_vi_vf
        self.N_vi_vf_target = N_vi_vf_target
        self.ghost_allocation = total_ghost_allocation
        
        infos = pd.DataFrame(columns=['vi','vf','N_vi_vf', 'N_vi_vf_target', 'ratio', 'ghost', 'ghost_ratio'])
        
        for vi_vf in N_vi_vf.keys():
            if N_vi_vf_target[vi_vf]-N_vi_vf[vi_vf] > 0:
                ghost_ratio = total_ghost_allocation[vi_vf]/(N_vi_vf_target[vi_vf]-N_vi_vf[vi_vf])
            else:
                ghost_ratio = np.nan
            infos.loc[infos.index.size] = [vi_vf[0],
                                           vi_vf[1],
                                           N_vi_vf_target[vi_vf]-N_vi_vf[vi_vf],
                                           N_vi_vf_target[vi_vf],
                                           (N_vi_vf_target[vi_vf]-N_vi_vf[vi_vf])/N_vi_vf_target[vi_vf],
                                           total_ghost_allocation[vi_vf],
                                           ghost_ratio]
            
        if sound > 0:
            print(infos)
        
        self.infos = infos
        
        if sound > 0:
            print('execution times')
            print(self.detailed_execution_time)
        
        return(map_f)
    
    def _sample(self, J, probability_maps):
        start_time = time.time()
        J = J.copy()
        # add P_vf__vi_z to J
        self._add_P_vf__vi_z_to_J(J, probability_maps, inplace=True)
        # print(J)
        # GART
        self.tested_pixels.append(J.index.size)
        self._generalized_acceptation_rejection_test(J, inplace=True, accepted_only=True)
                
        # sample all accepted kernels
        J = J.sample(frac=1, replace=False)
        
        self.detailed_execution_time['sampling'].append(time.time()-start_time)
        start_time = time.time()
        
        # patch surface
        if not self._draw_patches_parameters(J, list(probability_maps.layers.keys())):
            return(False)
        
        self.detailed_execution_time['patches_parameters_initialization'].append(time.time()-start_time)
                        
        J.reset_index(inplace=True)
        J.rename({'index':'j'}, level=0, axis=1, inplace=True)
        
        return(J)