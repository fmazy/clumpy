#!/usr/bin/env python3
# -*- coding: utf-8 -*-
       

class Patches():
    """
    Patch parameters object. Should prefer the child BootStrapPatch.

    Parameters
    ----------
    method : str
        Patch caracteristics draws method.
    """
    def draw(n):
        """
        draws patches.
        """
        return(True, True)
        
class BootstrapPatches(Patches):
    """
    Bootstrap patch parameters object.
    
    Parameters
    ----------
    areas : array-like of shape (n_patches,)
        Array of areas.
    eccentricities : array-like of shape (n_patches,)
        Array of eccentricities which correspond to areas.
    """
    def __init__(self, areas, eccentricities):
        self.areas = areas
        self.eccentricities = eccentricities

class TransitionPatches():
    """
    Patches informations for a given transition

    Parameters
    ----------
    isl_patches : BootstrapPatchParams, default=None
        The island patch parameters. Unexpected to be ``None`` if ``ratio>0``.
    exp_patches : BootstrapPatchParams, default=None
        The expansion patch parameters. Unexpected to be ``None`` if ``ratio<1``.
    isl_ratio : float
        The ratio of island areas over the whole areas to allocate.
    """
    def __init__(self,
                 isl_patches = None,
                 exp_patches = None,
                 isl_ratio = 1.0):
        
        if isl_ratio < 1.0 and exp_patches is None:
            raise(ValueError('Unexpected exp_patches for such a isl_ratio.'))
        if isl_ratio > 0.0 and isl_patches is None:
            raise(ValueError('Unexpected isl_patches for such a isl_ratio.'))
        
        self.isl_patches = isl_patches
        self.exp_patches = exp_patches
        self.isl_ratio = isl_ratio