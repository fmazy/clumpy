import numpy as np

from .._base._transition_matrix import TransitionMatrix

from ._allocator import Allocator, _update_P_v__Y_u
from ._gart import generalized_allocation_rejection_test

class UnbiasedMonoPixel(Allocator):
    def __init__(self,
                 verbose=0,
                 verbose_heading_level=1):
        self.verbose = verbose
        self.verbose_heading_level = verbose_heading_level

    def _allocate(self,
                  transition_matrix,
                  land,
                  lul_data,
                  lul_origin_data,
                  mask=None,
                  distances_to_states={},
                  path_prefix_transition_probabilities=None,
                  copy_geo=None):

        # check if it is really a land transition matrix
        transition_matrix._check_land_transition_matrix()

        J, P_v__u_Y, Y = land.transition_probabilities(
            transition_matrix=transition_matrix,
            lul=lul_data,
            mask=mask,
            distances_to_states=distances_to_states,
            path_prefix=path_prefix_transition_probabilities,
            copy_geo=copy_geo,
            return_Y=True)

        # GART
        V = generalized_allocation_rejection_test(P_v__u_Y,
                                                  transition_matrix.palette_v.get_list_of_values())

        # allocation !
        lul_data.flat[J] = V

