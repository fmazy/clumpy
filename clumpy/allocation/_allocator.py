"""
Allocators blabla.
"""

from .._base import State
from ..layer import LandUseLayer
from ..tools._path import path_split
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
                 calibrator=None,
                 verbose=0,
                 verbose_heading_level=1):
        self.calibrator = None
        self.verbose = verbose
        self.verbose_heading_level = verbose_heading_level

    def set_params(self,
                   **params):
        for key, param in params.items():
            setattr(self, key, param)

    def allocate(self,
                 lul, 
                 J,
                 P_v__u_Y,
                 final_states,
                 lul_origin=None,
                 distances_to_states={}):
        """
        Allocate

        Parameters
        ----------
        tm : TransitionMatrix
            Land transition matrix with only one state in ``tm.palette_u``.

        land : Land
            The studied land object.

        lul : LandUseLayer or ndarray
            The studied land use layer. If ndarray, the matrix is directly edited (inplace).

        lul_origin : LandUseLayer
            Original land use layer. Usefull in case of regional allocations. If ``None``, the  ``lul`` layer is copied.

        mask : MaskLayer, default = None
            The region mask layer. If ``None``, the whole map is studied.

        distances_to_states : dict(State:ndarray), default={}
            The distances matrix to key state. Used to improve performance.

        path : str, default=None
            The path to save result as a tif file.
            If None, the allocation is only saved within `lul`, if `lul` is a ndarray.
            Note that if ``path`` is not ``None``, ``lul`` must be LandUseLayer.

        path_transition_probabilities : str, default=None
            The path prefix to save transition probabilities

        Returns
        -------
        lul_allocated : LandUseLayer
            Only returned if ``path`` is not ``None``. The allocated map as a land use layer.
        """
        if isinstance(lul, LandUseLayer):
            lul_data = lul.get_data()
        else:
            lul_data = lul
        
        if lul_origin is None:
            lul_origin_data = lul_data.copy()
        else:
            lul_origin_data = lul_origin
        
        
        self._allocate(J,
                       P_v__u_Y,
                       final_states,
                       lul_data=lul_data,
                       lul_origin_data=lul_origin_data,
                       distances_to_states=distances_to_states)
        
        return(lul_data)
    
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
