import numpy as np
from tqdm import tqdm
from copy import deepcopy

from .._base._transition_matrix import TransitionMatrix
from ._allocator import Allocator, _update_P_v__Y_u
from ._gart import generalized_allocation_rejection_test as must
from ._patcher import _weighted_neighbors_patcher
from ..layer import LandUseLayer, RegionsLayer
from ..layer._proba_layer import create_proba_layer
from ..tools._console import title_heading

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
                 threshold_update_P_Y=0,
                 n_try=10 ** 3,
                 verbose=0):
        
        self.threshold_update_P_Y = threshold_update_P_Y
        self.n_try = n_try

        super().__init__(verbose=verbose)
    
    def allocate(self,
                 luc_layer:LandUseLayer,
                 initial_state,
                 final_states,
                 J,
                 Z,
                 tpe_func,
                 patches,
                 P_v,
                 regions_layer:RegionsLayer=None,
                 luc_layer_origin:LandUseLayer=None):
        """
        allocation. luc_layer_data and luc_layer_origin_data are ndarrays only.
        """
        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level) + 'Unbiased Allocation')
        
        if luc_layer_origin is None:
            luc_layer_origin = luc_layer.copy()
        
        P_v = P_v.copy()
        
        n_try = 0
        
        # for the first iteration, P_Y and P_Y__v are estimated
        P_Z__u = None
        P_Z__u_v = None
        
        keep_allocate = True
        
        n_used = 0
        n_used_max = len(J) * self.threshold_update_P_Y
        
        while keep_allocate and n_try < self.n_try:
            keep_allocate = False
            
            n_try += 1
            
            # P_v_patches = P_v.copy()
            # P_v_patches /= np.array([patch.areas.mean() for patch in patches])
            
            # compute transition probabilities
            # if P_Y and P_Y__v are None, they are estimated
            P_V__u_Z, P_Z__u, P_Z__u_v = tpe_func(Y=Z,
                                          P_v=P_v,
                                          P_Y = P_Z__u,
                                          P_Y__v = P_Z__u_v,
                                          return_P_Y__v=True,
                                          return_P_Y=True)
                        
            if n_try==1:
                proba_layer = create_proba_layer(J=J,
                                                 P=P_V__u_Z,
                                                 final_states=final_states,
                                                 shape=luc_layer.shape,
                                                 geo_metadata=luc_layer.geo_metadata)
            
            # pivot
            Q = must(P_V__u_Z)
            Q_selected = Q>=0
            J_pivot = J[Q_selected]
            Q_pivot = Q[Q_selected]
            
            i_shuffled = np.random.choice(a=Q_pivot.size, size=Q_pivot.size, replace=False)
            J_pivot = J_pivot[i_shuffled]
            Q_pivot = Q_pivot[i_shuffled]
            
            print('Q_pivot.shape', Q_pivot.shape)
                        
            # return(luc_layer)
                                    
            # convert P_v to a number of pixels
            P_v *= len(J)
            J_used = []
            
            print('start, rest ', P_v)
            
            for i in range(J_pivot.size):
                q = Q_pivot[i]
                final_state = final_states[q]
                s, J_used_i = patches[q].allocate(
                    lul=luc_layer,
                    lul_origin=luc_layer_origin,
                    j=J_pivot[i],
                    proba_layer=proba_layer.get_proba(final_state),
                    initial_state=initial_state,
                    final_state=final_state)
                
                if s > 0: # allocation succeed
                    P_v[q] -= s
                
                J_used += J_used_i
                n_used += len(J_used_i)
                
                if n_used >= n_used_max:
                    break
                
                if P_v[q] < 0:
                    P_v[q] = 0
                    break
            
            print('stop, rest ', P_v)
            
            # convert back P_v to a probability
            P_v /= len(J)
            
            # update J
            idx = ~np.isin(J, J_used)
            
            # if necessary, upate P_Y by setting it to None
            # else, the unused pixels are selected
            if n_used >= n_used_max:
                P_Z__u = None
                # set new threshold parameters
                # according to the new set of pixels
                n_used = 0
                n_used_max = idx.sum() * self.threshold_update_P_Y
            else:
                P_Z__u = P_Z__u[idx]
            
            # unused pixels are selected for P_Y__v
            # no updates are needed, the estimation is the same !
            P_Z__u_v = P_Z__u_v[idx]
            
            # finally, unused pixels are selected for J and X
            Z = Z[idx]
            J = J[idx]
            
            if P_v.max() > 0:
                keep_allocate = True
            
        return(luc_layer, proba_layer)