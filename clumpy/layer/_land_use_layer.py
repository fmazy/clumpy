# -*- coding: utf-8 -*-

import numpy as np

from ._layer import Layer
from ..tools._data import ndarray_suitable_integer_type

class LandUseLayer(Layer):
    """Define a Land Use Cover (LUC) layer.
    This layer can then used for the calibration stage or the allocation stage.
    
    Parameters
    ----------
    palette : Palette
        The states palette.
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


    Attributes
    ----------
    raster_ : :class:`rasterio.io.DatasetReader`
        The unbuffered data and metadata reader object, provided by :mod:`rasterio`.
    style_ : list of dict
        The style used for displaying.
    """
    
    def __init__(self,
                 palette,
                 label=None,
                 time=0,
                 path=None,
                 data=None,
                 copy_geo=None):

        if data is not None:
            # choose an appropriate dtype.
            data = ndarray_suitable_integer_type(data)
        
        super().__init__(label=label,
                         time=time,
                         path=path,
                         data=data,
                         copy_geo=copy_geo)
    
        self.palette = palette

    def __repr__(self):
        return(self.label)
        
    def set_palette(self, palette):
        """
        Set palette

        Parameters
        ----------
        palette : Palette
            The palette.

        Returns
        -------
        self : LandUseLayer
            The self object.

        """
        self.palette = palette
    
    def display(self,
                center,
                window,
                show=True,
                colorbar=True):
        """
        Display the land use cover layer through python console with matplotlib.

        Parameters
        ----------
        center : tuple of two integers
            Center position as a tuple.
        window : tuple of two integers
            Window dimensions as a tuple.
        show : bool, default=True
            If True, the ``plt.show()`` is applied.
        colorbar : bool, default=True
            If True, the colorbar is displayed.

        Returns
        -------
        plt : matplotlib.pyplot
            Pyplot object
        """
        
        ordered_palette = self.palette.sort(inplace=False)
        
        labels, values, colors = ordered_palette.get_list_of_labels_values_colors()
        
        colors = colors[:-1] + [colors[-2]] + [colors[-1]]
        bounds = np.array(values+[values[-1]+1])-0.5
        
        cmap = mpl_colors.ListedColormap(colors)
        norm = mpl_colors.BoundaryNorm(bounds, cmap.N)
        
        super().display(center=center,
                        window=window,
                        show=False,
                        colorbar=False,
                        interpolation='none',
                        cmap=cmap,
                        norm=norm)

        if colorbar:
            cb = plt.colorbar()
            cb.set_ticks(values)
            cb.set_ticklabels(labels)

        if show:
           plt.show()
        return(plt)
