# -*- coding: utf-8 -*-

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
    def __init__(self,
                 label=None,
                 time=0,
                 path=None,
                 data=None,
                 copy_geo=None):

        super().__init__(label=label,
                         time=time,
                         path=path,
                         data=data,
                         copy_geo=copy_geo)

        # if ~np.all(np.equal(np.unique(self.get_data()), np.array([0,1]))):
        #     raise(ValueError("Unexpected mask layer. Mask layer should be only composed by '0' and '1' values."))