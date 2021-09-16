#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 07:50:46 2021

@author: frem
"""

class PatchesParams():
    """
    Patches parameters object.

    Parameters
    ----------
    isl_patch : BootstrapPatchParams, default=None
        The island patch parameters. Unexpected to be ``None`` if ``ratio>0``.
    exp_patch : BootstrapPatchParams, default=None
        The expansion patch parameters. Unexpected to be ``None`` if ``ratio<1``.
    ratio : float
        The ratio of island areas over the whole areas to allocate.
    """
    def __init__(self,
                 isl_patch=None,
                 exp_patch=None,
                 ratio=1.0):
        self.isl_patch = isl_patch
        self.exp_patch = exp_patch
        self.ratio = ratio
        
        
class PatchParams():
    """
    Patch parameters object. Should prefer the child BootStrapPatch.

    Parameters
    ----------
    method : str
        Patch caracteristics draws method.
    """
    def __init__(self, method):
        self.method = method
        
class BootstrapPatchParams(PatchParams):
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
        
        super().__init__(method='bootstrap')