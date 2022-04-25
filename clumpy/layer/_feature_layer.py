# -*- coding: utf-8 -*-

from copy import deepcopy

from ._layer import Layer

class FeatureLayer(Layer):
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
                geo_metadata=None,
                bounded='none'):
        
        obj = super().__new__(cls, 
                              input_array,
                              label=label,
                              geo_metadata=geo_metadata)
        
        obj.bounded = bounded
        
        return obj  
    
    def copy(self):
        return FeatureLayer(np.array(self),
                            label=self.label,
                            geo_metadata=deepcopy(self.geo_metadata),
                            bounded=deepcopy(self.bounded))    