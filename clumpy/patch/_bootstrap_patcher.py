# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage
from skimage import measure

from ._patcher import Patcher, structures
from ..tools._data import np_drop_duplicates_from_column

class BootstrapPatcher(Patcher):
    """
    Bootstrap patch parameters object.

    Parameters
    ----------
    neighbors_structure : {'rook', 'queen'}, default='rook'
        The neighbors structure.

    avoid_aggregation : bool, default=True
        If ``True``, the patcher will avoid patch aggregations to respect expected patch areas.

    nb_of_neighbors_to_fill : int, default=3
        The patcher will allocate cells whose the number of allocated neighbors is greater than this integer
        (according to the specified ``neighbors_structure``)

    proceed_even_if_no_probability : bool, default=True
        The patcher will allocate even if the neighbors have no probabilities to transit.

    n_tries_target_sample : int, default=10**3
        Number of tries to draw samples in a biased way in order to approach the mean area.

    equi_neighbors_proba : bool, default=False
        If ``True``, all neighbors have the equiprobability to transit.
    """
    def __init__(self,
                 neighbors_structure = 'rook',
                 avoid_aggregation = True,
                 nb_of_missing_to_fill = 1,
                 proceed_even_if_no_probability = True,
                 n_tries_target_sample = 1000,
                 equi_neighbors_proba=False):
        
        super().__init__(neighbors_structure = neighbors_structure,
                         avoid_aggregation = avoid_aggregation,
                         nb_of_missing_to_fill = nb_of_missing_to_fill,
                         proceed_even_if_no_probability = proceed_even_if_no_probability,
                         n_tries_target_sample=n_tries_target_sample,
                         equi_neighbors_proba=equi_neighbors_proba)

    def _sample(self, n):
        idx = np.random.choice(self.areas.size, n, replace=True)
        
        return(self.areas[idx], self.eccentricities[idx])

    def set(self,
            areas,
            eccentricities):
        """
        Set areas and eccentricities.

        Parameters
        ----------
        areas : array-like of shape (n_patches,)
            Array of areas.
        eccentricities : array-like of shape (n_patches,)
            Array of eccentricities which correspond to areas.

        Returns
        -------
        self
        """

        if areas.size > 0 and areas.size == eccentricities.size:
            self.areas = areas
            self.eccentricities = eccentricities

            self.area_mean = np.mean(areas)
            self.eccentricities_mean = np.mean(eccentricities)

        return(self)

    def crop_areas(self,
                   min_area=-np.inf,
                   max_area=np.inf,
                   inplace=True):
        """
        Crop areas.

        Parameters
        ----------
        min_area : float, default=-np.inf
            Minimum area threshold.
        max_area : float, default=np.inf
            Maximum area threshold.

        Returns
        -------
        self
        """
        idx = (self.areas >= min_area) & (self.areas <= max_area)

        if inplace:
            self.areas = self.areas[idx]
            self.eccentricities = self.eccentricities[idx]
        else:
            return(BootstrapPatcher().set(areas=self.areas[idx],
                                        eccentricities=self.eccentricities[idx]))
        
    def fit(self,
            J,
            V_transited,
            shape):
        """Compute bootstrap patches
    
        Parameters
        ----------
        state : State
            The initial state of this land.
    
        final_states : [State]
            The final states list.
    
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
        
        try:
            structure = structures[self.neighbors_structure]
        
        except:
            raise (ValueError('ERROR : unexpected neighbors_structure value'))
    
        M = np.zeros(shape)
        M.flat[J[V_transited]] = 1

        lw, _ = ndimage.measurements.label(M, structure=structure)
        patch_id = lw.flat[J]
                
        # unique pixel for a patch
        one_pixel_from_patch = np.column_stack((J, patch_id))
        one_pixel_from_patch = np_drop_duplicates_from_column(one_pixel_from_patch, 1)

        one_pixel_from_patch = one_pixel_from_patch[1:, :]
        one_pixel_from_patch[:, 1] -= 1

        rpt = measure.regionprops_table(lw, properties=['area',
                                                        'inertia_tensor_eigvals'])

        self.areas = np.array(rpt['area'])
        
        l1_patch = np.array(rpt['inertia_tensor_eigvals-0'])
        l2_patch = np.array(rpt['inertia_tensor_eigvals-1'])

        self.eccentricities = np.zeros(self.areas.shape)
        id_none_mono_pixel_patches = self.areas > 1

        self.eccentricities[id_none_mono_pixel_patches] = 1 - np.sqrt(
            l2_patch[id_none_mono_pixel_patches] / l1_patch[id_none_mono_pixel_patches])

        # mono pixel patches are removed
        self.areas = self.areas[id_none_mono_pixel_patches]
        self.eccentricities = self.eccentricities[id_none_mono_pixel_patches]
                