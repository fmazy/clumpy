import numpy as np
from scipy import ndimage
from skimage import measure
from ..tools._data import np_drop_duplicates_from_column

from ._patch import BootstrapPatch

def compute_bootstrap_patches(state,
                              palette_v,
                              land,
                              lul_initial,
                              lul_final,
                              mask=None,
                              neighbors_structure='rook'):
    """Compute bootstrap patches

    Parameters
    ----------
    state : State
        The initial state of this land.

    palette_v : Palette
        The final palette.

    land : Land
        The studied land object.

    lul_initial : LandUseLayer
        The initial land use.

    lul_final : LandUseLayer
        The final land use.

    mask : MaskLayer, default = None
        The region mask layer. If ``None``, the whole area is studied.

    neighbors_structure : {'rook', 'queen'}, default='rook'
        The neighbors structure.

    Returns
    -------
    patches : dict(State:Patch)
        Dict of patches with states as keys.
    """
    if neighbors_structure == 'queen':
        structure = np.ones((3, 3))
    elif neighbors_structure == 'rook':
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]])
    else:
        raise (ValueError('ERROR : unexpected neighbors_structure value'))

    M_shape = lul_initial.get_data().shape

    patches = {}

    u = state.value

    J, V = land.get_values(lul_initial=lul_initial,
                           lul_final=lul_final,
                           mask=mask,
                           explanatory_variables=False)

    for state_v in palette_v:
        if state_v != state:
            # print(str(u) + ' -> ' + str(v))
            M = np.zeros(M_shape)
            M.flat[J[V == state_v.value]] = 1

            lw, _ = ndimage.measurements.label(M, structure=structure)
            patch_id = lw.flat[J]

            # unique pixel for a patch
            one_pixel_from_patch = np.column_stack((J, patch_id))
            one_pixel_from_patch = np_drop_duplicates_from_column(one_pixel_from_patch, 1)

            one_pixel_from_patch = one_pixel_from_patch[1:, :]
            one_pixel_from_patch[:, 1] -= 1

            rpt = measure.regionprops_table(lw, properties=['area',
                                                            'inertia_tensor_eigvals'])

            areas = np.array(rpt['area'])

            # return(patches, rpt)
            l1_patch = np.array(rpt['inertia_tensor_eigvals-0'])
            l2_patch = np.array(rpt['inertia_tensor_eigvals-1'])

            eccentricities = np.zeros(areas.shape)
            id_none_mono_pixel_patches = areas > 1

            eccentricities[id_none_mono_pixel_patches] = 1 - np.sqrt(
                l2_patch[id_none_mono_pixel_patches] / l1_patch[id_none_mono_pixel_patches])

            # mono pixel patches are removed
            areas = areas[id_none_mono_pixel_patches]
            eccentricities = eccentricities[id_none_mono_pixel_patches]

            patches[state_v] = BootstrapPatch().set(areas=areas,
                                                    eccentricities=eccentricities)

    return (patches)
