from ._allocator import Allocator
from ._gart import generalized_allocation_rejection_test
from ..layer import LandUseLayer, MaskLayer
from .._base._transition_matrix import TransitionMatrix

class UnbiasedMonoPixel(Allocator):
    def __init__(self,
                 calibrator=None,
                 verbose=0,
                 verbose_heading_level=1):
        
        super().__init__(calibrator=calibrator,
                         verbose=verbose,
                         verbose_heading_level=verbose_heading_level)
        
    def allocate(self,
                 J,
                 P,
                 final_states,
                 lul:LandUseLayer,
                 lul_origin:LandUseLayer=None,
                 mask:MaskLayer=None):
        """
        allocation. lul_data is ndarray only.
        """
        
        P, final_states = self.clean_proba(P=P, 
                                           final_states=final_states)
        
        # # GART
        V = generalized_allocation_rejection_test(P=P,
                                                  list_v=final_states)
        
        # # allocation !
        lul.flat[J] = V
        
        return(lul)

