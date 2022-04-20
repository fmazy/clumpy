# -*- coding: utf-8 -*-

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

    def __init__(self,
                 label=None,
                 time=0,
                 path=None,
                 data=None,
                 bounded = 'none',
                 copy_geo=None):
        
        super().__init__(label=label,
                         time=time,
                         path=path,
                         data=data,
                         copy_geo=copy_geo)
        
        self.bounded = bounded