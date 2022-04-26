"""
Allocators blabla.
"""
import numpy as np

from .._base import State
from .._base._transition_matrix import TransitionMatrix
from ..layer import LandUseLayer, MaskLayer
from ..calibration import Calibrator
from ..layer._proba_layer import create_proba_layer
from ..tools._path import path_split
from ._gart import generalized_allocation_rejection_test
from copy import deepcopy

class Allocator():
    """
    Allocator

    Parameters
    ----------
    verbose : int, default=0
        Verbosity level.

    verbose_heading_level : int, default=1
        Verbose heading level for markdown titles. If ``0``, no markdown title are printed.
    """

    def __init__(self,
                 calibrator:Calibrator=None,
                 verbose=0,
                 verbose_heading_level=1):
        self.calibrator = calibrator
        self.verbose = verbose
        self.verbose_heading_level = verbose_heading_level
    
    # def run(self,
    #         tm:TransitionMatrix,
    #         lul:LandUseLayer,
    #         features=None,
    #         lul_origin:LandUseLayer=None,
    #         mask:MaskLayer=None):
        
    #     if lul_origin is None:
    #         lul_origin = lul.copy()
    
    #     J, P, final_states = self.calibrator.transition_probabilities(
    #         lul=lul_origin,
    #         tm=tm,
    #         features=features,
    #         mask = mask,
    #         effective_transitions_only=False)
        
    #     P, final_states = self.clean_proba(P=P, 
    #                                        final_states=final_states)
        
    #     proba_layer = create_proba_layer(J=J,
    #                                      P=P,
    #                                      final_states=final_states,
    #                                      shape=lul.shape,
    #                                      geo_metadata=lul.geo_metadata)
        
    #     self.allocate(J=J,
    #                   P=P,
    #                   final_states=final_states,
    #                   lul=lul,
    #                   lul_origin=lul_origin,
    #                   mask=mask)
        
    #     return(lul, proba_layer)
        
    def _clean_proba(self, 
                    P, 
                    final_states):
        
        final_states = deepcopy(final_states)
        
        if self.calibrator.initial_state not in final_states:
            P = np.hstack((P, 1-P.sum(axis=1)[:,None]))
            final_states.append(self.calibrator.initial_state)
        
        return(P, final_states)
    
    def _sample_pivot(self,
                     J,
                     P,
                     final_states,
                     shuffle=True):
        V = generalized_allocation_rejection_test(P, 
                                                  final_states)

        print('V unique ', np.unique(V, return_counts=True))

        id_pivot = V != self.calibrator.initial_state
        V_pivot = V[id_pivot]
        J_pivot = J[id_pivot]
        
        if shuffle:
            n = J_pivot.size
            id_shuffled = np.random.choice(a=n, 
                                           size=n,
                                           replace=False)
            J_pivot = J_pivot[id_shuffled]
            V_pivot = V_pivot[id_shuffled]
        
        return(J_pivot, V_pivot)
    
    def set_params(self,
                   **params):
        for key, param in params.items():
            setattr(self, key, param)
    
def _update_P_v__Y_u(P_v__u_Y, tm, inplace=True):
    if not inplace:
        P_v__u_Y = P_v__u_Y.copy()

    tm._check_land_tm()

    state_u = tm.palette_u.states[0]
    id_state_u = tm.palette_v.get_id(state_u)

    # then, the new P_v is
    P_v__u_Y_mean = P_v__u_Y.mean(axis=0)
    multiply_cols = P_v__u_Y_mean != 0
    P_v__u_Y[:, multiply_cols] *= tm.M[0, multiply_cols] / P_v__u_Y_mean[
        multiply_cols]
    # set the closure with the non transition column
    P_v__u_Y[:, id_state_u] = 0.0
    P_v__u_Y[:, id_state_u] = 1 - P_v__u_Y.sum(axis=1)

    return (P_v__u_Y)
