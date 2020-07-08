from ._allocation import _Allocation, compute_P_vf__vi_from_transition_probability_maps
from ..calibration._calibration import _Calibration
from .. import definition
from ._patcher import _weighted_neighbors

import numpy as np
import pandas as pd
import time

class SimpleUnbiased(_Allocation):
    """Simple Unbiased Allocation Method
    
    Parameters
    ----------
    params : dict (default=None)
        the parameters dictionary
    """
    
    def __init__(self, params = None):
        super().__init__(params)
        
    def allocate_monopixel_patches(self,
                                   case:definition.Case,
                                   probability_maps=None,
                                   sound=2,
                                   dict_args={}):
        """
        Simple allocation of monopixels patches whithout scenario control.

        Parameters
        ----------
            
        case : definition.Case
            Starting case which have to be discretized.
            
        probability_maps : definition.TransitionProbabilityLayers (default=None)
            The transition probabilities maps.
            
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
            
            ``self.tested_pixels``
        """
        np.random.seed() # needed to seed in case of multiprocessing
        
        probability_maps=dict_args.get('probability_maps', probability_maps)
        sound=dict_args.get('sound', sound)
        
        start_time = time.time()
        J = case.discrete_J.copy()
            
        map_f_data = case.map_i.data.copy()
        
        self.execution_time['pixels_initialization']=time.time()-start_time
        start_time = time.time()
        
        try:
            self._add_P_vf__vi_z_to_J(J, probability_maps, inplace=True)
        except:
            raise TypeError('unexpected probability_maps')
            
        # GART
        self.tested_pixels = [J.index.size]
        self._generalized_acceptation_rejection_test(J, inplace=True, accepted_only=True)
        
        self.execution_time['sampling']=[time.time()-start_time]
        start_time = time.time()
                
        # allocation
        map_f_data.flat[J.index.values] = J.v.f.values
        
        self.execution_time['pixels_initialization']=time.time()-start_time

        self.execution_time['patches_parameters_initialization']=[0]
        
        # post processing
        map_f = definition.LandUseCoverLayer(name="luc_simple",
                                               time=None,
                                               scale=case.map_i.scale)
        map_f.import_numpy(data=map_f_data, sound=sound)
        
        if sound>0:
            print('FINISHED')
            print('========')
            print('execution times')
            print(self.execution_time)
        
            N_vi_vf = J.groupby([('v','i'), ('v','f')]).size().reset_index(name=('N_vi_vf', ''))
            print(N_vi_vf)
        
        return(map_f)
        
    def allocate(self,
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
        
        self.execution_time = {}
                
        start_time = time.time()
        P_vf__vi = compute_P_vf__vi_from_transition_probability_maps(case, probability_maps)
        
        self.execution_time['transition_matrix']=time.time()-start_time
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
        
        self.execution_time['pixels_initialization']=time.time()-start_time
        start_time = time.time()
        
        # select pivot-cells
        self.execution_time['sampling'] = []
        self.execution_time['patches_parameters_initialization'] = []
        
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
        
        self.execution_time['allocation']=time.time()-start_time - np.sum(self.execution_time['sampling']) - np.sum(self.execution_time['patches_parameters_initialization'])
                
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
            print(self.execution_time)
        
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
        
        self.execution_time['sampling'].append(time.time()-start_time)
        start_time = time.time()
        
        # patch surface
        if not self._draw_patches_parameters(J, list(probability_maps.layers.keys())):
            return(False)
        
        self.execution_time['patches_parameters_initialization'].append(time.time()-start_time)
                        
        J.reset_index(inplace=True)
        J.rename({'index':'j'}, level=0, axis=1, inplace=True)
        
        return(J)