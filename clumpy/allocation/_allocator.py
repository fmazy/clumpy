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

from scipy.stats import norm

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
    
    def nb_monte_carlo(self,
                       lul:LandUseLayer,
                       tm:TransitionMatrix,
                       features=None,
                       mask:MaskLayer=None,
                       alpha = 0.05,
                       epsilon = 0.001):
        
        if features is None:
            features = self.calibrator.features
                
        initial_state = self.calibrator.initial_state
        final_states = self.calibrator.tpe.get_final_states()
        
        final_states_id = {final_state:final_states.index(final_state) for final_state in final_states}
        P_v = np.array([tm.get(int(initial_state),
                               int(final_state)) for final_state in final_states])
        
        J = lul_origin.get_J(state=initial_state,
                      mask=mask)
        X = lul_origin.get_X(J=J, 
                             features=features)
        
        X = self.calibrator.feature_selector.transform(X)
        
        P, final_states, P_Y = self.calibrator.tpe.transition_probabilities(
            J=J,
            Y=X,
            P_v=P_v,
            return_P_Y=True,
            return_P_Y__v=False)
        
        p_alpha = norm.ppf(1-alpha/2)
        
        # remove initial state
        if initial_state in final_states:
            P = np.delete(P, list(final_states).index(initial_state), axis=1)
        
        return np.max(p_alpha**2/epsilon**2 * P * (1-P) / (P_Y * P.shape[0]))
    
    def _clean_proba(self, 
                    P, 
                    final_states):
        
        
        P = P.copy()
        final_states = deepcopy(final_states)
        
        if self.calibrator.initial_state in final_states:
            idx = final_states.index(int(self.calibrator.initial_state))
            P[:, idx] = 1-np.delete(P, idx, axis=1).sum(axis=1)
        else:  
            final_states.append(self.calibrator.initial_state.value)
            P = np.hstack((P, 1-P.sum(axis=1)[:,None]))
        
        return(P, final_states)
    
    def _sample_pivot(self,
                     J,
                     P,
                     final_states,
                     shuffle=True):
        
        P, final_states = self._clean_proba(P=P, 
                                            final_states=final_states)
        
        V = generalized_allocation_rejection_test(P, 
                                                  final_states)

        id_pivot = V != int(self.calibrator.initial_state)
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
