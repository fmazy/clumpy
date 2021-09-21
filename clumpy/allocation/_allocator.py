"""
Allocators blabla.
"""

from ._patch import Patch
from .._base import LandUseLayer, State
from ..tools._path import path_split

class Allocator():
    """
    Allocator
    """
    def __init__(self,
                 verbose=0):
        self.patches = {}
        self.verbose = verbose

    def add_patch(self,
                  state,
                  patch):
        """
        Add a patch object for a given final state.

        Parameters
        ----------
        state : State
            The final state.
        patch : Patch
            The corresponding Patch object.

        Returns
        -------
        self : Land
            The self object.
        """

        if not isinstance(patch, Patch):
            raise(TypeError("Unexpected 'patch'. A clumpy.allocation.Patch object is expected."))
        if not isinstance(state, State):
            raise(TypeError("Unexpected 'state'. A clumpy.State object is expected."))
        self.patches[state] = patch

        return(self)

    def set_patches(self,
                    patches):
        """
        Set patches

        Parameters
        ----------
        patches : dict(State:Patch)
            Dict of patches with states as keys.

        Returns
        -------
        self
        """
        _check_patches(patches)
        self.patches = patches

        return(self)

    def allocate(self,
                 transition_matrix,
                 land,
                 lul,
                 lul_origin=None,
                 mask=None,
                 distances_to_states={},
                 path=None):
        """
        Allocate

        Parameters
        ----------
        transition_matrix : TransitionMatrix
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

        Returns
        -------
        lul_allocated : LandUseLayer
            Only returned if ``path`` is not ``None``. The allocated map as a land use layer.
        """

        # check if it is really a land transition matrix
        transition_matrix._check_land_transition_matrix()

        if lul_origin is None:
            lul_origin = lul

        if isinstance(lul_origin, LandUseLayer):
            lul_origin_data = lul_origin.get_data()
        else:
            lul_origin_data = lul_origin

        if isinstance(lul, LandUseLayer):
            lul_data = lul.get_data().copy()
        else:
            lul_data = lul

        self._allocate(transition_matrix=transition_matrix,
                   land=land,
                   lul_data=lul_data,
                   lul_origin_data=lul_origin_data,
                   mask=mask,
                   distances_to_states=distances_to_states)

        if path is not None:
            folder_path, file_name, file_ext = path_split(path)
            return (LandUseLayer(label='file_name',
                                 data=lul_data,
                                 copy_geo=lul_origin,
                                 path=path,
                                 palette=lul_origin.palette))

def _check_patches(patches):
    if not isinstance(patches, dict):
        raise("Unexpected 'patches' type. A dict(State:Patch) is expected.")
    for state, patch in patches.items():
        if not isinstance(state, State) or not isinstance(patch, Patch):
            raise("Unexpected 'patches' type. A dict(State:Patch) is expected.")
