# -*- coding: utf-8 -*-

from copy import deepcopy

from ._layer import Layer

class MaskLayer(Layer):
    """
    Mask layer.

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
    copy_geo : :class:`LandUseLayer`, default=None
        The layer from whose geo metadata are copied.
        If ``None``, geo metadata are set to ``None``.
    """
    
    def __new__(cls, 
                input_array,
                label=None,
                geo_metadata=None):
        
        obj = super().__new__(cls, 
                              input_array,
                              label=label,
                              geo_metadata=geo_metadata)
        
        
        return obj  
    
    def copy(self):
        return MaskLayer(np.array(self),
                         label=self.label,
                         geo_metadata=deepcopy(self.geo_metadata))