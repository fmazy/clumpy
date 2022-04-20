import numpy as np

from .._base._transition_matrix import TransitionMatrix

from ._allocator import Allocator, _update_P_v__Y_u
from ._gart import generalized_allocation_rejection_test

class UnbiasedMonoPixel(Allocator):
    def __init__(self,
                 verbose=0,
                 verbose_heading_level=1):
        
        super().__init__(verbose=verbose,
                         verbose_heading_level=verbose_heading_level)

    def _allocate(self,
                  lul_data,
                  p):
        
        for region_label, p_region in p.items():
            for initial_state, p_land in p_region.items():
                J, P_v__u_Y, final_states = p_land
                # GART
                V = generalized_allocation_rejection_test(P=P_v__u_Y,
                                                          list_v=final_states)
                                
                # allocation !
                lul_data.flat[J] = V

