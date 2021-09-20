from ._patch import Patch
from .._base import LandUseLayer, State
from ..tools._path import path_split

class Allocator():
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

        if ~isinstance(patch, Patch):
            raise(TypeError("Unexpected 'patch'. A clumpy.allocation.Patch object is expected."))
        if ~isinstance(state, State):
            raise(TypeError("Unexpected 'state'. A clumpy.State object is expected."))
        self.patches[state] = patch

        return(self)

    def set_patches(self,
                    patches):
        _check_patches(patches)
        self.patches = patches

    def allocate(self,
                 state,
                 land,
                 P_v,
                 palette_v,
                 luc,
                 luc_origin=None,
                 mask=None,
                 distances_to_states={},
                 path=None):

        if luc_origin is None:
            luc_origin = luc

        if isinstance(luc_origin, LandUseLayer):
            luc_origin_data = luc_origin.get_data()
        else:
            luc_origin_data = luc_origin

        if isinstance(luc, LandUseLayer):
            luc_data = luc.get_data().copy()
        else:
            luc_data = luc

        self._allocate(state=state,
                   land=land,
                   P_v=P_v,
                   palette_v=palette_v,
                   luc_data=luc_data,
                   luc_origin_data=luc_origin_data,
                   mask=mask,
                   distances_to_states=distances_to_states)

        if path is not None:
            folder_path, file_name, file_ext = path_split(path)
            return (LandUseLayer(label='file_name',
                                 data=luc_data,
                                 copy_geo=luc_origin,
                                 path=path,
                                 palette=luc_origin.palette))

def _check_patches(patches):
    if ~isinstance(patches, dict):
        raise("Unexpected 'patches' type. A dict(State:Patch) is expected.")
    for state, patch in patches.items():
        if ~isinstance(state, State) or ~isinstance(patch, Patch):
            raise("Unexpected 'patches' type. A dict(State:Patch) is expected.")
