from ._allocation import _Allocation
from ..calibration._calibration import _Calibration
from .. import definition
from ._patcher import _weighted_neighbors

import pandas as pd
import numpy as np
import time

class GeneralizedVonNeumann(_Allocation):
    """Generalized Von Neumann Allocation Method
    
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
                                   probability_maps = None,
                                   sound=2,
                                   dict_args={}):
        """
        Simple allocation of monopixels patches whithout scenario control.

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
            
            ``self.execution_time``
            
            ``self.tested_pixels``
        """
        np.random.seed() # needed to seed in case of multiprocessing
        
        P_vf__vi=dict_args.get('P_vf__vi', P_vf__vi)
        probability_maps=dict_args.get('probability_maps', probability_maps)
        sound=dict_args.get('sound', sound)
        
        start_time = time.time()
        
        if type(probability_maps) == type(None):
            if type(P_vf__vi) == type(None):
                P_vf__vi = calibration.P_vf__vi
            
            probability_maps = calibration.transition_probability_maps(case, P_vf__vi)
        else:
            probability_maps = probability_maps.copy()
        
        self.execution_time['transition_probability_maps']=time.time()-start_time
        start_time = time.time()
        
        map_f_data = case.map_i.data.copy()
        J = case.discrete_J.copy()
        J.fillna(0, inplace=True) # the sampling function does not accept any NaN features
        
        self.execution_time['pixels_initialization']=time.time()-start_time
        
        # get pivot cells
        J_pivotcells = self._sample(J, probability_maps, draw_patches_parameters=False, random_sample=False)
        
        J_pivotcells.set_index('j', inplace=True)
        
        # allocation
        start_time = time.time()
        map_f_data.flat[J_pivotcells.index.values] = J_pivotcells.v.f.values
        
        self.execution_time['allocation']=time.time()-start_time

        self.execution_time['patches_parameters_initialization']=[0]
        
        # post processing
        map_f = definition.LandUseCoverLayer(name="luc_simple",
                                   time=None,
                                   scale=case.map_i.scale)
        map_f.import_numpy(data=map_f_data, sound=sound)
        
        if sound>1:
            print('FINISHED')
            print('========')
            print('execution times')
            print(self.execution_time)
            
            N_vi_vf = J_pivotcells.groupby([('v','i'), ('v','f')]).size().reset_index(name=('N_vi_vf', ''))
            print(N_vi_vf)
        
        return(map_f)
        
    def allocate(self,
                 case:definition.Case,
                 calibration=None,
                 P_vf__vi=None,
                 probability_maps=None,
                 update='none',
                 sound=2,
                 dict_args={}):
        """
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
            
            ``self.execution_time``
                One microseconds precision for Linux and Mac OS and 16 milliseconds precision for Windows
            
            ``self.tested_pixels``
            
            ``self.infos``
            
            ``self.N_vi_vf``
            
            ``self.N_vi_vf_target``
        """
        np.random.seed() # needed to seed in case of multiprocessing
        
        P_vf__vi=dict_args.get('P_vf__vi', P_vf__vi)
        probability_maps=dict_args.get('probability_maps', probability_maps)
        update=dict_args.get('update', update)
        sound=dict_args.get('sound', sound)
        
        if calibration == None:
            calibration = _Calibration()
        
        start_time = time.time()
        
        if type(probability_maps) == type(None):
            if type(P_vf__vi) == type(None):
                P_vf__vi = calibration.P_vf__vi
            
            probability_maps = calibration.transition_probability_maps(case, P_vf__vi)
        else:
            probability_maps = probability_maps.copy()
        
        self.execution_time['transition_probability_maps']=time.time()-start_time
        start_time = time.time()
        
        map_f_data = case.map_i.data.copy()
        J = case.discrete_J.copy()
        J.fillna(0, inplace=True) # the sampling function does not accept any NaN features
                
        N_vi_vf = self._compute_N_vi_vf(list(probability_maps.layers.keys()), J, P_vf__vi)
        N_vi_vf_target = N_vi_vf.copy()
        if sound >1:
            print(N_vi_vf)
        
        # build the P_z__vi map 
        calibration.compute_P_z__vi(case)
        M_P_z__vi = calibration.build_P_z__vi_map(case)
        
        self.execution_time['pixels_initialization']=time.time()-start_time
        start_time = time.time()
        
        # get pivot cells
        self.execution_time['sampling'] = []
        self.execution_time['patches_parameters_initialization'] = []
        self.tested_pixels = []
        
        J_pivotcells = self._sample(J, probability_maps, draw_patches_parameters=True, random_sample=True)
        
        # allocation
        # sum_N_vi_vf_target = np.sum(list(N_vi_vf_target.values()))
        
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
                if sound>1:
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
                    if sound>1:
                        print('ghost threshold reached for ('+str(vi)+','+str(vf)+')')
                    update_P_z__vi_trigger = True
                    ghost_allocation[(vi,vf)] = 0
            
            # on met à jour les probabilités
            if update_P_z__vi_trigger:
                if sound>1:
                    print('P_z__vi update...')
                # on met à jour J_initial en retirant les pixels qui ont transités
                J[('v','f')] = map_f_data.flat[J.index.values]
                J.drop(J.loc[J.v.i != J.v.f].index.values, axis=0, inplace=True)
                # on calcule le nouveau P_z__vi
                # !!! le discrete J doit être mis à jour avec
                # J qui vient d'être recalculé...
                # build the P_z__vi map 
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
                J_pivotcells = self._sample(J, probability_maps, draw_patches_parameters=True, random_sample=True)
                # le compteur idx est réinitialisé
                idx = -1
                if sound>1:
                    print('done')
        
        self.execution_time['allocation']=time.time()-start_time - np.sum(self.execution_time['sampling']) - np.sum(self.execution_time['patches_parameters_initialization'])
        
        
        
        # post processing
        map_f = definition.LandUseCoverLayer(name="luc_simple",
                                   time=None,
                                   scale=case.map_i.scale)
        map_f.import_numpy(data=map_f_data, sound=sound)
        if sound>1:
            print('FINISHED')
            print('========')
        
        # some informations are saved
        self.N_vi_vf = N_vi_vf
        self.N_vi_vf_target = N_vi_vf_target
        self.ghost_allocation = total_ghost_allocation
        
        infos = pd.DataFrame(columns=['vi','vf','N_vi_vf', 'N_vi_vf_target', 'ratio', 'ghost', 'ghost_ratio'])
        
        for vi_vf in N_vi_vf.keys():
            infos.loc[infos.index.size] = [vi_vf[0],
                                           vi_vf[1],
                                           N_vi_vf_target[vi_vf]-N_vi_vf[vi_vf],
                                           N_vi_vf_target[vi_vf],
                                           (N_vi_vf_target[vi_vf]-N_vi_vf[vi_vf])/N_vi_vf_target[vi_vf],
                                           total_ghost_allocation[vi_vf],
                                           total_ghost_allocation[vi_vf]/(N_vi_vf_target[vi_vf]-N_vi_vf[vi_vf])]
        if sound>1:
            print(infos)
        
        self.infos = infos
        if sound>1:
            print('execution times')
            print(self.execution_time)
        
        return(map_f)
    
    def _sample(self, J, probability_maps, draw_patches_parameters=False, random_sample=False):
        print('\t sample...')
        start_time = time.time()
        J = J.copy()
        
        self._add_P_vf__vi_z_to_J(J, probability_maps, inplace=True)
        
        J.reset_index(drop=False, inplace=True)
        J.rename({'index':'j'}, level=0, axis=1, inplace=True)
        
        J_pivotcells = pd.DataFrame()
        for vi in J.v.i.unique():            
            J_vi = J.loc[J.v.i==vi]
            
            N_vi_z = J_vi.groupby(J_vi[['z']].columns.to_list()).size().reset_index(name=('N_vi_z', ''))
            
            J_vi_unique = J_vi[['z', 'P_vf__vi_z']].drop_duplicates().copy().merge(N_vi_z)
                        
            J_vi_unique['N_to_draw'] = np.round(J_vi_unique.P_vf__vi_z.sum(axis=1) * J_vi_unique.N_vi_z)
            J_vi_unique['id_z'] = np.arange(J_vi_unique.index.size)
                        
            J_vi = J_vi.merge(right=J_vi_unique, how='left')
            
            # J_vi.set_index('j', inplace=True)
                        
            J_vi_unique.set_index('id_z', inplace=True)
            
            for id_z in range(J_vi_unique.index.size):
                J_pivotcells = pd.concat([J_pivotcells, J_vi.loc[J_vi.id_z == id_z].sample(n=int(J_vi_unique.loc[id_z].N_to_draw))])
                                   
        self.tested_pixels.append(J_pivotcells.index.size)
        
        if J_pivotcells.index.size == 0:
            return(J_pivotcells)
        
        #  P_vf__vi_z changes to respect sum P_vf__vi_z = 1                
        J_pivotcells.P_vf__vi_z = J_pivotcells.P_vf__vi_z.divide(J_pivotcells.P_vf__vi_z.sum(axis=1), axis="index").values
        
        # GART
        self._generalized_acceptation_rejection_test(J_pivotcells, inplace=True, accepted_only=True)    
        
        if random_sample:
            J_pivotcells = J_pivotcells.sample(frac=1, replace=False)
        
        self.execution_time['sampling'].append(time.time()-start_time)
        start_time = time.time()
        
        # patch surface
        if draw_patches_parameters:
            if not self._draw_patches_parameters(J_pivotcells, list_vi_vf=list(probability_maps.layers.keys())):
                print('draw patch parameters error')
                return(False)
            
            self.execution_time['patches_parameters_initialization'].append(time.time()-start_time)
            
        J_pivotcells.reset_index(drop=True, inplace=True)
        
        print('\t done')
        
        return(J_pivotcells)