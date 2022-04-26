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
                 threshold_update_P_Y=0,
                 n_try=10 ** 3,
                 verbose=0,
                 verbose_heading_level=1):
        
        self.threshold_update_P_Y = threshold_update_P_Y
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
        
        # for the first iteration, P_Y and P_Y__v are estimated
        P_Y = None
        P_Y__v = None
        
        keep_allocate = True
        
        n_used = 0
        n_used_max = len(J) * self.threshold_update_P_Y
        
        while keep_allocate and n_try < self.n_try:
            keep_allocate = False
            
            n_try += 1
            
            P_v_patches = P_v.copy()
            P_v_patches /= self.calibrator.patchers.area_mean(final_states=final_states)
            
            # compute transition probabilities
            # if P_Y and P_Y__v are None, they are estimated
            P, final_states, P_Y, P_Y__v = self.calibrator.tpe.transition_probabilities(
                J=J,
                Y=X,
                P_v=P_v_patches,
                P_Y=P_Y,
                P_Y__v=P_Y__v,
                return_P_Y=True,
                return_P_Y__v=True)
            
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
                        
            # convert P_v to a number of pixels
            P_v *= len(J)
            J_used = []
            for i in range(J_pivot.size):
                final_state = V_pivot[i]
                s, J_used_i = self.calibrator.patchers[final_state].allocate(
                    lul=lul,
                    lul_origin=lul_origin,
                    j=J_pivot[i])
                
                if s > 0: # allocation succeed
                    P_v[final_states_id[final_state]] -= s
                else:
                    keep_allocate = True
                
                J_used += J_used_i
                n_used += len(J_used_i)
                
                if n_used >= n_used_max:
                    break
            
            # convert back P_v to a probability
            P_v /= len(J)
            
            # update J
            idx = ~np.isin(J, J_used)
            
            # if necessary, upate P_Y by setting it to None
            # else, the unused pixels are selected
            if n_used >= n_used_max:
                P_Y = None
                # set new threshold parameters
                # according to the new set of pixels
                n_used = 0
                n_used_max = idx.sum() * self.threshold_update_P_Y
                keep_allocate = True
            else:
                P_Y = P_Y[idx]
            
            # unused pixels are selected for P_Y__v
            # no updates are needed, the estimation is the same !
            P_Y__v = P_Y__v[idx]
            
            # finally, unused pixels are selected for J and X
            X = X[idx]
            J = J[idx]
            
        return(lul, proba_layer)