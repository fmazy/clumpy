import numpy as np

from .._base._layer import LandUseLayer

from ._allocator import Allocator
from ._gart import generalized_allocation_rejection_test
from ._patcher import _weighted_neighbors_patcher
from ..tools._path import path_split

class Unbiased(Allocator):
    def __init__(self,
                 update_P_v__u_Y = True,
                 n_allocation_try = 10**3,
                 verbose=0):

        self.update_P_v__u_Y = update_P_v__u_Y
        self.n_allocation_try = n_allocation_try

        super().__init__(verbose = verbose)

    def _allocate(self,
                   state,
                   land,
                   P_v,
                   palette_v,
                   luc_data,
                   luc_origin_data,
                   mask=None,
                   distances_to_states={}):
        """
        allocation. luc can be both LandUseLayer and ndarray.

        Parameters
        ----------
        state : TYPE
            DESCRIPTION.
        P_v : TYPE
            DESCRIPTION.
        palette_v : TYPE
            DESCRIPTION.
        luc : TYPE
            DESCRIPTION.
        luc_origin : TYPE, optional
            DESCRIPTION. The default is None.
        mask : TYPE, optional
            DESCRIPTION. The default is None.
        distances_to_states : TYPE, optional
            DESCRIPTION. The default is {}.
        path : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        # ghost initialization
        palette_v_without_u = palette_v.remove(state)
        n_ghost = {state_v: 1 for state_v in palette_v_without_u}

        # P_v initialization
        P_v = P_v.copy()

        # get the id of the initial state
        id_state = palette_v.get_id(state)

        # first_len_J initialization
        first_len_J = None

        J_used = []

        n_try = 0
        while np.sum(list(n_ghost.values())) > 0 and n_try < self.n_allocation_try:
            n_try += 1

            # P_v is divided by patch area mean
            # P_v_patches is then largely smaller.
            # one keep P_v to update it after the allocation try.
            P_v_patches = P_v.copy()
            P_v_patches[id_state] = 1
            for id_state_v, state_v in enumerate(palette_v):
                if state_v != state:
                    P_v_patches[id_state_v] /= self.patches[state_v].area_mean
                    P_v_patches[id_state] -= P_v[id_state_v]

            # if P_v__u_Y has to be updated
            # or if it is the first loop
            # compute P(v|u,Y)
            if self.update_P_v__u_Y or n_try == 1:
                luc_data_P_v__u_Y_update = luc_data.copy()
                luc_data_P_v__u_Y_update.flat[J_used] = -1
                J, P_v__u_Y = land._compute_tpe(state=state,
                                                luc=luc_data_P_v__u_Y_update,
                                                P_v=P_v_patches,
                                                palette_v=palette_v,
                                                mask=mask,
                                                distances_to_states=distances_to_states)
            else:
                # else just update J by removing used pixels.
                # it is faster but implies a bias.
                idx_to_keep = ~np.isin(J, J_used)
                J = J[idx_to_keep]
                P_v__u_Y = P_v__u_Y[idx_to_keep]

            # save the original number of pixels in this land
            # it is used to edit P(v)
            if first_len_J is None:
                first_len_J = len(J)

            # Allocation try
            # results are : all pixels used, number of allocation for each state
            # and numbre of ghost pixels for each states.
            J_used_last, n_allocated, n_ghost = self._try_allocate(state=state,
                                                                     land=land,
                                                                     J=J,
                                                                     P_v__u_Y=P_v__u_Y,
                                                                     palette_v=palette_v,
                                                                     luc_origin_data=luc_origin_data,
                                                                     luc_data=luc_data)

            # the pixel used and which are now useless are saved in J_used
            J_used += J_used_last

            # P_v update : for the next loop, one need less pixels
            P_v[id_state] = 1
            for id_state_v, state_v in enumerate(palette_v):
                if state_v != state:
                    P_v[id_state_v] -= n_allocated[state_v] / first_len_J
                    P_v[id_state] -= P_v[id_state_v]

    def _try_allocate(self,
                        state,
                        land,
                        J,
                        P_v__u_Y,
                        palette_v,
                        luc_origin_data,
                        luc_data):
        """
        Try to allocate

        Parameters
        ----------
        state : TYPE
            DESCRIPTION.
        J : TYPE
            DESCRIPTION.
        P_v__u_Y : TYPE
            DESCRIPTION.
        palette_v : TYPE
            DESCRIPTION.
        luc_origin_data : TYPE
            DESCRIPTION.
        luc_data : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        palette_v_without_u = palette_v.remove(state)

        # GART
        V = generalized_allocation_rejection_test(P_v__u_Y, palette_v.get_list_of_values())

        id_pivot = V != state.value
        V_pivot = V[id_pivot]
        J_pivot = J[id_pivot]

        areas = {}
        eccentricities = {}
        eccentricity_mean = {}
        eccentricity_std = {}

        for state_v, patch in self.patches.items():
            j = V_pivot == state_v.value
            if j.size > 0:
                areas[state_v], eccentricities[state_v] = patch.target_sample(j.sum())

                eccentricity_mean[state_v] = np.mean(eccentricities[state_v])
                eccentricity_std[state_v] = np.std(eccentricities[state_v])

        P_v__u_Y_maps = {}
        for id_v, state_v in enumerate(palette_v):
            P_v__u_Y_maps[state_v] = P_v__u_Y[:, id_v]

        J_used = []

        n_allocated = {state_v: 0 for state_v in palette_v_without_u}
        n_ghost = {state_v: 0 for state_v in palette_v_without_u}

        # allocation
        for id_j in np.random.choice(J_pivot.size, J_pivot.size, replace=False):
            v = V_pivot[id_j]
            state_v = palette_v._get_by_value(v)
            allocated_area, J_used_last = _weighted_neighbors_patcher(map_i_data=luc_origin_data,
                                                                      map_f_data=luc_data,
                                                                      map_P_vf__vi_z=P_v__u_Y_maps[state_v],
                                                                      j_kernel=J_pivot[id_j],
                                                                      vi=state.value,
                                                                      vf=v,
                                                                      patch_S=areas[state_v][id_j],
                                                                      eccentricity_mean=eccentricity_mean[state_v],
                                                                      eccentricity_std=eccentricity_std[state_v],
                                                                      neighbors_structure=self.patches[
                                                                          state_v].neighbors_structure,
                                                                      avoid_aggregation=self.patches[
                                                                          state_v].avoid_aggregation,
                                                                      nb_of_neighbors_to_fill=self.patches[
                                                                          state_v].nb_of_neighbors_to_fill,
                                                                      proceed_even_if_no_probability=self.patches[
                                                                          state_v].proceed_even_if_no_probability)

            J_used += J_used_last

            n_allocated[state_v] += allocated_area

            # if the allocation has been aborted
            if allocated_area == 0:
                n_ghost[state_v] += 1

        return (J_used, n_allocated, n_ghost)
