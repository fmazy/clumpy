import numpy as np

from .._base._transition_matrix import TransitionMatrix

from ._allocator import Allocator, _update_P_v__Y_u
from ._gart import generalized_allocation_rejection_test

class UnbiasedMonoPixel(Allocator):
    def __init__(self,
                 calibrator=None,
                 verbose=0,
                 verbose_heading_level=1):
        
        super().__init__(calibrator=calibrator,
                         verbose=verbose,
                         verbose_heading_level=verbose_heading_level)

    def _allocate(self,
                  J,
                  P_v__u_Y,
                  final_states,
                  lul_data,
                  **kwargs):
        
        # GART
        V = generalized_allocation_rejection_test(P=P_v__u_Y,
                                                  list_v=final_states)
        
        print((V==2).mean())
        
        # allocation !
        lul_data.flat[J] = V

