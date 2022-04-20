"""
Allocators blabla.
"""

from .._base import LandUseLayer, State
from ..tools._path import path_split


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
                 verbose=0,
                 verbose_heading_level=1):
        self.verbose = verbose
        self.verbose_heading_level = verbose_heading_level

    def set_params(self,
                   **params):
        for key, param in params.items():
            setattr(self, key, param)

    def allocate(self,
                 calibrator,
                 tm,
                 lul,
                 lul_origin=None,
                 mask=None,
                 distances_to_states={},
                 path=None,
                 path_transition_probabilities=None,
                 copy_geo=None):
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

        path_prefix_transition_probabilities : str, default=None
            The path prefix to save transition probabilities

        Returns
        -------
        lul_allocated : LandUseLayer
            Only returned if ``path`` is not ``None``. The allocated map as a land use layer.
        """

        # check if it is really a land transition matrix
        tm._check_land_tm()

        if lul_origin is None:
            lul_origin = lul

        if isinstance(lul_origin, LandUseLayer):
            lul_origin_data = lul_origin.get_data()
            copy_geo = lul_origin
        else:
            lul_origin_data = lul_origin

        if isinstance(lul, LandUseLayer):
            lul_data = lul.get_data().copy()
        else:
            lul_data = lul

        self._allocate(tm=tm,
                       land=land,
                       lul_data=lul_data,
                       lul_origin_data=lul_origin_data,
                       mask=mask,
                       distances_to_states=distances_to_states,
                       path_prefix_transition_probabilities=path_prefix_transition_probabilities,
                       copy_geo=copy_geo)

        if path is not None:
            folder_path, file_name, file_ext = path_split(path)
            return (LandUseLayer(label='file_name',
                                 data=lul_data,
                                 copy_geo=lul_origin,
                                 path=path,
                                 palette=lul_origin.palette))

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
