# -*- coding: utf-8 -*-

from copy import deepcopy

from ._layer import Layer
import numpy as np 

class EVLayer(Layer):
    """Define a feature layer.
    This layer can then used for the calibration stage or the allocation stage.
    
    Parameters
    ----------
    label : str, default=``None``
        The layer label.
    time : float, default=``0``
        The layer time (year unit).
    path : str
        The tiff file path.
        If ``data`` is provided, a new tiff file will be created according to this path
        and this operation overwrites the file path if exists.
    data : :class:`numpy.ndarray`, defaul=None
        The data to write. If ``None``, no writing is made.
    
    bounded : {'none', 'left', 'right', 'both'}, default:'none'
        Boundary trigger.
    
    copy_geo : :class:`LandUseLayer`, default=None
        The layer from whose geo metadata are copied.
        If ``None``, geo metadata are set to ``None``.

    Attributes
    ----------
    raster_ : :class:`rasterio.io.DatasetReader`
        The unbuffered data and metadata reader object, provided by :mod:`rasterio`.
    """

    def __new__(cls, 
                input_array,
                label=None,
                dtype=None,
                geo_metadata=None,
                bounded='none'):
        
        obj = super().__new__(cls, 
                              input_array,
                              label=label,
                              dtype=dtype,
                              geo_metadata=geo_metadata)
        
        obj.bounded = bounded
        
        return obj  
    
    def copy(self):
        return EVLayer(np.array(self),
                       label=None,
                       geo_metadata=deepcopy(self.geo_metadata),
                       bounded=deepcopy(self.bounded))    
    
def get_bounds(evs):
    """
    Returns bounds list of given EV.

    Parameters
    ----------
    evs : list of EVLayer
        Explanatory variables given as a list.

    Returns
    -------
    bounds : list of string
        Bounds list in the same order as evs.

    """
    bounds = []
    for ev in evs:
        if isinstance(ev, EVLayer):
            bounds.append(ev.bounded)
        elif type(ev) is int:
            bounds.append('left')
    
    return bounds