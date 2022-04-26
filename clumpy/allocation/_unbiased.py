import numpy as np
from tqdm import tqdm
from copy import deepcopy

from .._base._transition_matrix import TransitionMatrix
from ._allocator import Allocator, _update_P_v__Y_u
from ._gart import generalized_allocation_rejection_test
from ._patcher import _weighted_neighbors_patcher
from ..layer import LandUseLayer, MaskLayer
from ..layer._proba_layer import create_proba_layer


class Unbiased(Allocator):
    """
    Unbiased method.

    Parameters
    ----------
    update_P_Y : bool, default=True
        If ``True``, P(Y) is updated at each iteration.

    n_allocation_try : int, default=10**3
        Maximum number of iterations.

    verbose : int, default=0
        Verbosity level.

    verbose_heading_level : int, default=1
        Verbose heading level for markdown titles. If ``0``, no markdown title are printed.
    """

    def __init__(self,
                 calibrator=None,
                 update_P_Y=True,
                 n_try=10 ** 3,
                 verbose=0,
                 verbose_heading_level=1):
        
        self.update_P_Y = update_P_Y
        self.n_try = n_try

        super().__init__(calibrator=calibrator,
                         verbose=verbose,
                         verbose_heading_level=verbose_heading_level)

    def allocate(self,
                 lul:LandUseLayer,
                 tm:TransitionMatrix,
                 features=None,
                 lul_origin:LandUseLayer=None,
                 mask:MaskLayer=None):
        """
        allocation. lul_data and lul_origin_data are ndarrays only.
        """
        
        if lul_origin is None:
            lul_origin = lul.copy()
        
        if features is None:
            features = self.calibrator.features
        
        tm = tm.extract(self.calibrator.initial_state)
        
        initial_state = self.calibrator.initial_state
        final_states = deepcopy(self.calibrator.final_states)
        
        
        
        n_try = 0
        
        J = lul_origin.get_J(state=initial_state,
                      mask=mask)
        X = lul_origin.get_X(J=J, 
                             features=features)
        
        X = self.calibrator.feature_selector.transform(X)
        
        P_Y = None
        P_Y__v = None
        
        keep_allocate = True
        while keep_allocate and n_try < self.n_try:
            n_try += 1
            
            tm_patches = tm.patches(patchers=self.calibrator.patchers,
                                    inplace=False)
            
            P_v = tm_patches.M[0,:]
            
            P, final_states, P_Y, P_Y__v = self.calibrator.tpe.transition_probabilities(
                J=J,
                Y=X,
                P_v=P_v,
                P_Y=P_Y,
                P_Y__v=P_Y__v,
                return_P_Y=True,
                return_P_Y__v=True)
            
            if not self.update_P_Y:
                P_Y = None
            
            P, final_states = self._clean_proba(P=P, 
                                                final_states=final_states)
            
            if n_try==1:
                proba_layer = create_proba_layer(J=J,
                                                 P=P,
                                                 final_states=final_states,
                                                 shape=lul.shape,
                                                 geo_metadata=lul.geo_metadata)
            
            # pivot
            J_pivot, V_pivot = self._sample_pivot(J=J, 
                                                  P=P, 
                                                  final_states=final_states,
                                                  shuffle=True)
            
            # try allocate
            tm.M *= len(J)
            J_allocated = []
            J_ghost = []
            for i in range(J_pivot.size):
                s, J_allocated_i = self.calibrator.patchers[V_pivot[i]].allocate(lul=lul,
                                                                                 lul_origin=lul_origin,
                                                                                 j=J_pivot[i])
                if s > 0: # allocation succeed
                    # tm.set()
                    # update tm
                    J_allocated += J_allocated_i
                else:
                    J_ghost += J_allocated_i
            
            if len(J_ghost) == 0:
                keep_allocate = False
            else:
                # update tm
                # _reduce_tm(tm,
                #             expected_allocated,
                #             n_allocated,
                #             n_idx_J_unused)
        
        return(lul, proba_layer)
    
    def _try_allocate(self,
                      J_pivot,
                      V_pivot,
                      lul):

        
        
        return(lul)
        
    def archive(self):
        # ghost initialization
        n_ghost = {final_state: 0 for final_state in final_states}
        
        # get the id of the initial_state
        id_initial_state = palette_v.get_id(initial_state)

        cols_v = np.delete(np.arange(len(palette_v)), id_initial_state)

        # J_used may have redundant cells indices
        J_used = []

        if self.verbose > 0:
            print('Allocation start...')
            print('P_v : ' + str(tm.M[0]))
            print('update_P_Y : ' + str(self.update_P_Y) + '\n')

        n_try = 0
        while np.sum(list(n_ghost.values())) > 0 and n_try < self.n_allocation_try:
            n_try += 1

            # tm is divided by patch area mean
            # The new one is then largely smaller.
            # one keep tm to update it after the allocation try.
            tm_patches = tm.patches(patches=self.patches,
                                    inplace=False)

            # if P_v__u_Y has to be updated
            # or if it is the first loop
            # compute P(v|u,Y)
            if n_try == 1:
                J, P_v__u_Y, final_initial_states, Y = self.calibrator.transition_probabilities(
                    lul=lul_data,
                    tm=tm_patches,
                    features=None,
                    mask=None,
                    distances_to_initial_states={},
                    effective_transitions_only=True,
                    save_P_Y__v=True,
                    save_P_Y=~self.update_P_Y,
                    return_Y=True)

                n_idx_J_unused = len(J)
                idx_J_unused = np.arange(n_idx_J_unused)

            else:
                P_v__u_Y = land.transition_probability_estimator.transition_probability(
                    tm=tm_patches,
                    Y=Y,
                    id_J=idx_J_unused,
                    compute_P_Y__v=False,
                    compute_P_Y=self.update_P_Y,
                    save_P_Y__v=False,
                    save_P_Y=False)

            expected_allocated = {initial_state_v: n_idx_J_unused * tm.M[0, id_initial_state_v] for id_initial_state_v, initial_state_v
                                  in
                                  enumerate(palette_v)}

            # Allocation try
            # results are : all pixels used, number of allocation for each initial_state
            # and numbre of ghost pixels for each initial_states.
            J_used_last, n_allocated, n_ghost = self._try_allocate(initial_state=initial_state,
                                                                   J=J[idx_J_unused],
                                                                   P_v__u_Y=P_v__u_Y,
                                                                   palette_v=palette_v,
                                                                   lul_origin_data=lul_origin_data,
                                                                   lul_data=lul_data)

            # the pixel used and which are now useless are saved in J_used
            J_used += J_used_last

            if self.verbose > 0:
                print('try #' + str(n_try) + ' - n_allocated : ' + str(n_allocated) + ' - sum(n_ghost) : ' + str(
                    np.sum(list(n_ghost.values()))))

            # update J by removing used pixels.
            idx_J_unused = ~np.isin(J, J_used)
            n_idx_J_unused = np.sum(idx_J_unused)
            if n_idx_J_unused == 0:
                print('/!\\ No more pixels available for allocation /!\\')
                break

            # update the transition matrix
            _reduce_tm(tm,
                       expected_allocated,
                       n_allocated,
                       n_idx_J_unused)
    
    
    
    def _try_allocate2(self,
                      initial_state,
                      J,
                      P_v__u_Y,
                      palette_v,
                      lul_origin_data,
                      lul_data):
        """
        Try to allocate
        """
        palette_v_without_u = palette_v.remove(initial_state)

        print('gart parameters', P_v__u_Y.sum(axis=0).astype(int), palette_v.get_list_of_values())
        # GART
        V = generalized_allocation_rejection_test(P_v__u_Y, palette_v.get_list_of_values())

        print('V unique ', np.unique(V, return_counts=True))

        id_pivot = V != initial_state
        V_pivot = V[id_pivot]
        J_pivot = J[id_pivot]

        # patcher parameters initialization
        areas = np.ones(J_pivot.size)
        eccentricities = np.ones(J_pivot.size)
        eccentricity_mean = {}
        eccentricity_std = {}
        P_v__u_Y_maps = {}

        # for each final initial_state
        for id_v, initial_state_v in enumerate(palette_v):
            # if it is a real transition
            if initial_state_v != initial_state:
                # count the number of pivot cells
                j = V_pivot == initial_state_v.value
                n_j = np.sum(j)
                # if their is at least one cell
                if n_j > 0:
                    # if initial_state_v is a patches key :
                    if initial_state_v in self.patches.keys():
                        # sample areas and eccentricities
                        areas[j], eccentricities[j] = self.patches[initial_state_v].target_sample(n_j)

                        # compute area and eccentricity means
                        eccentricity_mean[initial_state_v] = np.mean(eccentricities[j])
                        eccentricity_std[initial_state_v] = np.std(eccentricities[j])
                        # if this initial_state_v has not been patches calibrated.
                        # by default, all areas are set to 1 pixel.

                    # generate P_v__u_Y maps, used by _weighted_neighbors_patcher
                    P_v__u_Y_maps[initial_state_v] = np.zeros(lul_data.shape)
                    P_v__u_Y_maps[initial_state_v].flat[J] = P_v__u_Y[:, id_v]

        J_used = []

        n_allocated = {initial_state_v: 0 for initial_state_v in palette_v_without_u}
        n_ghost = {initial_state_v: 0 for initial_state_v in palette_v_without_u}

        id_j_sampled = np.random.choice(J_pivot.size, J_pivot.size, replace=False)

        if self.verbose > 0:
            print('start patcher loop')
            id_j_sampled = tqdm(id_j_sampled)

        # allocation
        for id_j in id_j_sampled:

            v = V_pivot[id_j]
            initial_state_v = palette_v._get_by_value(v)
            allocated_area, J_used_last = _weighted_neighbors_patcher(map_i_data=lul_origin_data,
                                                                      map_f_data=lul_data,
                                                                      map_P_vf__vi_z=P_v__u_Y_maps[initial_state_v],
                                                                      j_kernel=J_pivot[id_j],
                                                                      vi=initial_state,
                                                                      vf=v,
                                                                      patch_S=areas[id_j],
                                                                      eccentricity_mean=eccentricity_mean[initial_state_v],
                                                                      eccentricity_std=eccentricity_std[initial_state_v],
                                                                      neighbors_structure=self.patches[
                                                                          initial_state_v].neighbors_structure,
                                                                      avoid_aggregation=self.patches[
                                                                          initial_state_v].avoid_aggregation,
                                                                      nb_of_neighbors_to_fill=self.patches[
                                                                          initial_state_v].nb_of_neighbors_to_fill,
                                                                      proceed_even_if_no_probability=self.patches[
                                                                          initial_state_v].proceed_even_if_no_probability,
                                                                      equi_neighbors_proba=self.patches[
                                                                          initial_state_v].equi_neighbors_proba)

            J_used += J_used_last

            n_allocated[initial_state_v] += allocated_area

            # if the allocation has been aborted
            if allocated_area == 0:
                n_ghost[initial_state_v] += 1

        return (J_used, n_allocated, n_ghost)


def _reduce_tm(tm,
               expected_allocated,
               n_allocated,
               n_idx_J_unused):
    initial_state = tm.palette_u.initial_states[0]
    palette_v = tm.palette_v
    # get the id of the initial initial_state
    id_initial_state = palette_v.get_id(initial_state)

    tm.M[0, id_initial_state] = 1
    for id_initial_state_v, initial_state_v in enumerate(palette_v):
        if initial_state_v != initial_state:
            expected_allocated[initial_state_v] -= n_allocated[initial_state_v]
            if expected_allocated[initial_state_v] < 0:
                expected_allocated[initial_state_v] = 0
            tm.M[0, id_initial_state_v] = expected_allocated[initial_state_v] / n_idx_J_unused
            tm.M[0, id_initial_state] -= tm.M[0, id_initial_state_v]
