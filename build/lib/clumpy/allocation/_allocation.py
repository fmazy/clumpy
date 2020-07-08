"""
intro
"""

from scipy import ndimage # for the fill holes function
from scipy import stats
import numpy as np
import pandas as pd
import tqdm
import time
import scipy # for signal -> convolution, linalg -> matrix operations
import multiprocessing as mp
# import pandas as pd

# from ..definition.data import create_J
from ..calibration._calibration import _Calibration


class _Allocation():
    def __init__(self, params=None):
        self.params = params
        
        self.execution_time = {'sampling':[], 'patches_parameters_initialization':[]}
        self.tested_pixels = []
        
    def _draw_patches_parameters(self, J, list_vi_vf):
        J['S_patch'] = 0
        for key in list_vi_vf:
            vi = key[0]
            vf = key[1]
            for isl_exp in ['isl','exp']:
                if isl_exp in self.params[(vi,vf)]['patches_parameters'].keys():
                    patches_parameters = self.params[(vi,vf)]['patches_parameters'][isl_exp]['area']
                    if patches_parameters['type'] == 'normal':
                        J.loc[(J.v.i==vi) & (J.v.f==vf),'S_patch'] = np.round(np.random.normal(loc=patches_parameters['mean'],
                                                                                        scale=patches_parameters['sd'],
                                                                                        size=J.loc[(J.v.i==vi) & (J.v.f==vf)].index.size))
                    elif patches_parameters['type'] == 'lognormal':
                        mu = np.log(patches_parameters['mean'])-0.5*np.log(1+patches_parameters['variance']/patches_parameters['mean']**2)
                        sigma = np.sqrt(np.log(1+patches_parameters['variance']/patches_parameters['mean']**2))
                        J.loc[(J.v.i==vi) & (J.v.f==vf),'S_patch'] = np.round(np.random.lognormal(mean=mu,
                                                                                        sigma=sigma,
                                                                                        size=J.loc[(J.v.i==vi) & (J.v.f==vf)].index.size)    )
                    else:
                        print('ERROR during "draw_patches_parameters" : vi', vi, 'vf', vf, 'isl_exp', isl_exp, ' -- unexpected distribution type')
                        return(False)
                # else:
                    # print('ERROR during "draw_patches_parameters" : vi', vi, 'vf', vf, ' -- ', isl_exp, ' was expected')
                    # return(False)
        return(True)
    
    def _compute_N_vi_vf(self, list_vi_vf, J, P_vf__vi):
        N_vi_vf = {}
        
        for key in list_vi_vf:
            vi = key[0]
            vf = key[1]
            N_vi_vf[(vi,vf)] = int(P_vf__vi.loc[P_vf__vi.v.i==vi].P_vf__vi[vf].values[0] * J.loc[J.v.i==vi].index.size)+1
        
        return(N_vi_vf)
    
    def _generalized_acceptation_rejection_test(self, J, inplace=False, accepted_only=False):
        """
        """
        # cum sum columns
        # first column creation
        
        if not inplace:
            J = J.copy()
            
        # vf column
        J[('v','f')] = J.v.i
        
        if 'P_vf__vi_z' in J.columns:
            list_vf = J.P_vf__vi_z.columns.to_list()
            for vf in list_vf:
                J[('P_vf__vi_z_cs', vf)] = 0
            # then cum sum
            J[[('P_vf__vi_z_cs', vf) for vf in list_vf]] = np.cumsum(J[['P_vf__vi_z']].values, axis=1)
            # random value
            J['gart_x'] = np.random.random(J.index.size)
        
            # vf attribution
            for vf in list_vf[::-1]:
                J.loc[J.gart_x<J[('P_vf__vi_z_cs',vf)], ('v','f')] = vf
            
            # drop columns
            J.drop(['P_vf__vi_z_cs', 'gart_x'], axis=1, level=0, inplace=True)
        
        if accepted_only:
            J.drop(J.loc[J.v.i == J.v.f].index.values, axis=0, inplace=True)
        
        if not inplace:
            return(J)
        
    def _add_P_vf__vi_z_to_J(self, J, probability_maps, inplace=False):
        if not inplace:
            J = J.copy()
        # probability_building 
            
        for key, pm in probability_maps.layers.items():
            vi = key[0]
            vf = key[1]
            # probability
            if ('P_vf__vi_z', vf) not in J.columns.to_list():
                J[('P_vf__vi_z', vf)] = 0
            J.loc[J.v.i==vi, [('P_vf__vi_z', vf)]] = pm.data.flat[J.loc[J.v.i==vi].index.values]
            
        # check probability integrity, ie sum for vf P(vf|vi,z)<=1
        if J.P_vf__vi_z.sum(axis=1).max() > 1:
            print('WARNING : probability error, the sum is > 1 for almost one pixel : ', J.P_vf__vi_z.sum(axis=1).max())
            
        if not inplace:
            return(J)
        
    # def _create_J_restricted_to_vi_with_transitions(self, map_i, probability_maps):
    #     # get all vi
    #     vi_list = []
    #     for k in probability_maps.layers.keys():
    #         if k[0] not in vi_list:
    #             vi_list.append(k[0])
            
    #     # all pixels
    #     J = create_J(map_i)
        
    #     # restrict to vi with transitions
    #     J = J.loc[J.v.i.isin(vi_list)].copy()
        
    #     return(J)

    def _get_maps_f_vi_vf_neighbors(self, list_vi_vf, map_f_data, neighbors_structure='rook'):
        """
        not used anywhere
        """
        
        # on veut le nombre de voisins vf pour chaque pixel vi
        # l'idée c'est donc d'appliquer une convolution bien choisie à J_vf
        # et de n'extraire que les pixels de J_vi. avec ça ça devrait le faire
        
        maps_f_vi_vf_neighbors = {}
        
        for vi_vf in list_vi_vf:
            vi = vi_vf[0]
            vf = vi_vf[1]
            
            # on pourrait optimiser et ne pas calculer deux fois la même carte...
            map_vi = (map_f_data == vi).astype(int)  
            map_vf = (map_f_data == vf).astype(int)
            
            if neighbors_structure == 'queen':
                convolution_mask = np.array([[1,1,1],
                                             [1,0,1],
                                             [1,1,1]])
            elif neighbors_structure == 'rook':
                convolution_mask = np.array([[0,1,0],
                                             [1,0,1],
                                             [0,1,0]])
            else:
                print('ERROR: unexpected neighbors structure in get_maps_f_vi_vf_neighbors')
            # calcul du nombre de voisins
            map_vf_convoluted = scipy.signal.convolve2d(map_vf,convolution_mask,'same')
        
            maps_f_vi_vf_neighbors[(vi,vf)] = map_vf_convoluted * map_vi
        
        return(maps_f_vi_vf_neighbors)


    
    def _get_J_with_vf_dist_constraint(J, luc_data, Ti, P_vf__vi_z_names, isl_exp):
        """
        not used anywhere
        """
    
        J_with_vf_dist_constraint = J.copy()
        
        for vf in P_vf__vi_z_names.keys():
            # on prend la carte de distance à vf:
            # if sound_level > 0:
            #     print('\t '+str(Tif)+'...')
            dist = ndimage.distance_transform_edt(1-(luc_data == vf).astype(int))
            J_with_vf_dist_constraint['d'] = dist.flat[J_with_vf_dist_constraint.index.values]
            
            # constraint
            if isl_exp == 'exp':
                J_with_vf_dist_constraint.loc[J_with_vf_dist_constraint.d>1,
                                                 P_vf__vi_z_names[vf]] = 0
            elif isl_exp == 'isl':
                J_with_vf_dist_constraint.loc[J_with_vf_dist_constraint.d <= Ti.Tif[vf].param_allocation['isl_min_dist'],
                                                 P_vf__vi_z_names[vf]] = 0
                            
        # il s'agit maintenant de ne garder que les pixels dont au moins une proba est > 0
        # pour ça on somme les probas et on récupère uniquement les pixels dont la somme est >0
        J_with_vf_dist_constraint['P_sum'] = J_with_vf_dist_constraint[list(P_vf__vi_z_names.values())].sum(axis=1)
        J_with_vf_dist_constraint = J_with_vf_dist_constraint.loc[J_with_vf_dist_constraint.P_sum > 0]
        
        J_with_vf_dist_constraint.drop(['d', 'P_sum'], axis=1, inplace=True)
        
        return(J_with_vf_dist_constraint)
        
    def test(self,
             monopixel_patches = True,
             alpha=0.95,
             epsilon=0.01,
             gamma=0.8,
             cores=1,
             **kwargs):
        """
        Tests the allocation according to the calibration. Given the law of large numbers, if one repeats the allocation a certain amount of times, one should find the calibration by simple frequency analysis.

        Parameters
        ----------
        monopixel_patches : Boolean (default=True)
            If ``True``, allocates only mono-pixel patches without any other considerations. If ``False``, multipixel patches will be designed. The model should therefore have been detailed with allocation parameters. This kind of multipixel patches test requires of curse a significant computation time whereas the monopixel one.
        alpha : Float (default=0.95)
            Signifiance level of the test which gives a threshold value for p. The more ``alpha`` closed to ``1``, the more the test is significant.
        epsilon : Float (default=0.01)
            The probability inference precision in case of an unbiased model.
        gamma : Float (defualt=0.8)
            The features combinaisons selection according to :math:`\sum_z P(z|v_i,v_f)`. It is deeply recommended to have ``gamma<1``.
        cores : Int (default=1)
            Number of cores used for parallel computations.
        \**kwargs : ?
            remaining parameters used by ``self.allocate`` or ``self.allocate_monopixel_patches`` methods.
        Returns
        -------
        J : pandas DataFrame
            The observed allocations frequencies according to the features.

        """       
        
        case=kwargs.get('case', None)
        probability_maps=kwargs.get('probability_maps', None)
        
        calibration = _Calibration()
        
        # restrict J to vi and z columns
        # il faut tester ici si il y a une colonne z dans discrete_J !
        J = case.discrete_J.copy()[[('v', 'i')]+case.discrete_J[['z']].columns.to_list()].fillna(0)
        N_vi_z = J.groupby([('v', 'i')]+J[['z']].columns.to_list()).size().reset_index(name=('N_vi_z', ''))
        J.drop_duplicates(inplace=True)
        J = J.reset_index().merge(N_vi_z, how='left').set_index('index')
                
        # get probability maps
        
        
        
        
        # add probabilities to J
        self._add_P_vf__vi_z_to_J(J, probability_maps, inplace=True)
                
        p_alpha = abs(stats.norm.interval(alpha)[0])
        
        P_z__vi = calibration.compute_P_z__vi(case, output='return')
        
        J = J.reset_index()
        J = J.merge(P_z__vi, how='left')
        
        P_vf__vi = compute_P_vf__vi_from_transition_probability_maps(case, probability_maps)
        J = J.merge(P_vf__vi, how='left')

        J = J.set_index('index')
        
        for vf in J.P_vf__vi_z.columns.to_list():
            # on détermine les pixels qui rentrent en considération.
            # d'abord calcul des P_z__vi_vf (ils ne sont pas naifs !)
            J['P_z__vi_vf'] = J.P_vf__vi_z[vf] * J.P_z__vi / J.P_vf__vi[vf]
            # on s'assure que leur somme donne bien 1 malgré les biais de calibration
            J.P_z__vi_vf /= J.P_z__vi_vf.sum()
            # on les trie dans l'ordre décroissant
            J.sort_values(by='P_z__vi_vf', ascending=False, inplace=True)
            # les idx sélectionnés sont ceux qui permettent d'atteindre gamma
            idx = J.P_z__vi_vf.cumsum() <= gamma
            
            J.loc[idx,('N', vf)] = (1-J.loc[idx, ('P_vf__vi_z', vf)])*p_alpha**2/(epsilon**2*J.loc[idx, ('P_vf__vi_z', vf)]*J.loc[idx, ('N_vi_z', '')])
            
        J.drop(['P_z__vi_vf', 'P_vf__vi', 'P_z__vi'], level=0, axis=1, inplace=True)
        
        
        
        N = int(J.N.max().max())
        
        J.drop('N', level=0, axis=1, inplace=True)
        
        print('N=', N)
        
        time.sleep(0.5)
        
        J.drop(('N_vi_z',''), level=0, axis=1, inplace=True)
        
        cols = [('v', 'i'), ('v','f')]+J[['z']].columns.to_list()+[('total_N_vi_vf_z', '')]
        cols = pd.MultiIndex.from_tuples(cols)
        
        if cores > 1:
            pool = mp.Pool(cores)
            sub_N_vi_vf_z = []
            sub_N = int(N/cores)+1
            sub_N_vi_vf_z = pool.starmap(self._call_allocation_N_times, [(case, sub_N, monopixel_patches, cols, i, kwargs) for i in range(cores)])
            pool.close()
        else:
            self._call_allocation_N_times(case,
                                          N,
                                          monopixel_patches,
                                          cols,
                                          0,
                                          kwargs)
        
                
        for i in range(cores-1):
            sub_N_vi_vf_z[0] = sub_N_vi_vf_z[0].merge(sub_N_vi_vf_z[i+1], how='outer', on=[('v', 'i'), ('v','f')]+J[['z']].columns.to_list(), suffixes=('','_toadd'))
            sub_N_vi_vf_z[0].total_N_vi_vf_z = sub_N_vi_vf_z[0].total_N_vi_vf_z.add(sub_N_vi_vf_z[0].total_N_vi_vf_z_toadd, fill_value=0)
            sub_N_vi_vf_z[0].drop('total_N_vi_vf_z_toadd', level=0, axis=1, inplace=True)
        
        N_vi_vf_z = sub_N_vi_vf_z[0]
        
        N_vi_vf_z = N_vi_vf_z.merge(N_vi_z, how='left')
        N_vi_vf_z.total_N_vi_vf_z = N_vi_vf_z.total_N_vi_vf_z.div(N_vi_vf_z.N_vi_z*N, fill_value=0)
        
        J.reset_index(inplace=True)
        for vf in J.P_vf__vi_z.columns.to_list():
            N_vi_vf_z[('post_P_vf__vi_z',vf)] = 0
            N_vi_vf_z.loc[N_vi_vf_z.v.f==vf, ('post_P_vf__vi_z',vf)] = N_vi_vf_z.total_N_vi_vf_z
            
            J = J.merge(N_vi_vf_z.loc[N_vi_vf_z.v.f==vf, [('v','i')]+N_vi_vf_z[['z']].columns.to_list()+[('post_P_vf__vi_z',vf)]], how='left')
            J['post_P_vf__vi_z', vf] = J['post_P_vf__vi_z', vf].fillna(0)
            N_vi_vf_z.drop(('post_P_vf__vi_z', vf), level=0, axis=1, inplace=True)
        
        J.set_index('index', inplace=True)
        
        return(J)

    def _call_allocation_N_times(self, case, N, monopixel_patches, cols, core, dict_args):
        J_simu = pd.DataFrame(columns=cols)
        
        if core ==0:
            items = tqdm.tqdm(range(N))
        else:
            items = range(N)
        
        for i in items:
            if monopixel_patches:
                map_f = self.allocate_monopixel_patches(case = case,
                                                        dict_args=dict_args,
                                                        sound=0)
            else:
                map_f = self.allocate(case = case,
                                      dict_args = dict_args,
                                      sound=0)
            
            J_f = case.discrete_J.copy()[['v','z']].fillna(0)
            J_f[('v','f')] = map_f.data.flat[J_f.index.values]
            
            
            # on retire ceux qui n'ont pas transité
            J_f.drop(J_f.loc[J_f.v.i == J_f.v.f].index.values, axis=0, inplace=True)
                        
            N_vi_vf_z = J_f.groupby([('v', 'i'), ('v','f')]+J_f[['z']].columns.to_list()).size().reset_index(name=('N_vi_vf_z', ''))
            
            J_simu = J_simu.merge(N_vi_vf_z, how='outer')

            J_simu.total_N_vi_vf_z = J_simu.total_N_vi_vf_z.add(J_simu.N_vi_vf_z, fill_value=0)
            
            J_simu.drop(('N_vi_vf_z',''), axis=1, inplace=True)
        
        return(J_simu)

def compute_P_vf__vi_from_transition_probability_maps(case, probability_maps):
    """
    returns P_vf__vi based on transition parobability maps and a given case.

    Parameters
    ----------
    case : Case
        case.
    probability_maps : definition.TransitionProbabilityLayers
        transition probability layers.

    Returns
    -------
    P_vf__vi : pandas DataFrame
        Transition matrix as a pandas DataFrame

    """
    
    # how many unique vf ? and vi ?
    list_vf = []
    list_vi = []
    for vi_vf in probability_maps.layers.keys():
        if vi_vf[0] not in list_vi:
            list_vi.append(vi_vf[0])
        if vi_vf[1] not in list_vf:
            list_vf.append(vi_vf[1])
    
    cols = [('v', 'i')] + [('P_vf__vi',vf) for vf in list_vf]
    cols = pd.MultiIndex.from_tuples(cols)
    P_vf__vi = pd.DataFrame(columns=cols)
    
    P_vf__vi[('v','i')] = list_vi
    
    for vi_vf, transition_probability_layer in probability_maps.layers.items():
        vi = vi_vf[0]
        vf = vi_vf[1]
        
        P_vf__vi.loc[P_vf__vi.v.i==vi, ('P_vf__vi',vf)] = transition_probability_layer.data.sum()/case.J.index.size

    return(P_vf__vi)
