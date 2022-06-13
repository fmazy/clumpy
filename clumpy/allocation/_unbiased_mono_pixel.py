import numpy as np

from ._allocator import Allocator
from ._gart import generalized_allocation_rejection_test
from ..layer import LandUseLayer, MaskLayer
from .._base._transition_matrix import TransitionMatrix
from ..layer._proba_layer import create_proba_layer
from ..tools._console import title_heading

class UnbiasedMonoPixel(Allocator):
    def __init__(self,
                 calibrator=None,
                 verbose=0,
                 verbose_heading_level=1):
        
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
        allocation. lul_data is ndarray only.
        """
        
        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level) + 'Unbiased Allocation')
        
        if lul_origin is None:
            lul_origin = lul.copy()
        
        if features is None:
            features = self.calibrator.features
                
        initial_state = self.calibrator.initial_state
        final_states = self.calibrator.tpe.get_final_states()
        
        final_states_id = {final_state:final_states.index(final_state) for final_state in final_states}
        P_v = np.array([tm.get(int(initial_state),
                               int(final_state)) for final_state in final_states])
                
        n_try = 0
        
        J = lul_origin.get_J(state=initial_state,
                      mask=mask)
        X = lul_origin.get_X(J=J, 
                             features=features)
        
        X = self.calibrator.feature_selector.transform(X)
        
        P, final_states = self.calibrator.tpe.transition_probabilities(
            J=J,
            Y=X,
            P_v=P_v,
            P_Y=None,
            P_Y__v=None,
            return_P_Y=False,
            return_P_Y__v=False)
        
        proba_layer = create_proba_layer(J=J,
                                         P=P,
                                         final_states=final_states,
                                         shape=lul.shape,
                                         geo_metadata=lul.geo_metadata)
        
        # # GART
        V = generalized_allocation_rejection_test(P=P,
                                                  list_v=final_states)
        
        # # allocation !
        lul.flat[J] = V
        
        return(lul, proba_layer)

