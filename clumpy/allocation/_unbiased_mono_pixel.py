import numpy as np

from .._base._tm import TransitionMatrix

from ._allocator import Allocator, _update_P_v__Y_u
from ._gart import generalized_allocation_rejection_test

class UnbiasedMonoPixel(Allocator):
    def __init__(self,
                 verbose=0,
                 verbose_heading_level=1):
        
        super().__init__(verbose=verbose,
                         verbose_heading_level=verbose_heading_level)

    def _allocate(self,
                  tm,
                  land,
                  lul_data,
                  lul_origin_data,
                  mask=None,
                  distances_to_states={},
                  path_prefix_transition_probabilities=None,
                  copy_geo=None):

        # check if it is really a land transition matrix
        tm._check_land_tm()

        J, P_v__u_Y, Y = land.transition_probabilities(
            tm=tm,
            lul=lul_data,
            mask=mask,
            distances_to_states=distances_to_states,
            path_prefix=path_prefix_transition_probabilities,
            copy_geo=copy_geo,
            return_Y=True)

        # GART
        V = generalized_allocation_rejection_test(P_v__u_Y,
                                                  tm.palette_v.get_list_of_values())

        # allocation !
        lul_data.flat[J] = V

