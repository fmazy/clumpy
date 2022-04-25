# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage
from copy import deepcopy

from ._layer import Layer
from matplotlib import colors as mpl_colors
from matplotlib import pyplot as plt

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
    
    def __new__(cls, 
                input_array,
                label=None,
                geo_metadata=None):
        
        obj = super().__new__(cls, 
                              input_array,
                              label=label,
                              geo_metadata=geo_metadata)
        
        obj.distances = {}
        
        return obj  

    def copy(self):
        return LandUseLayer(np.array(self),
                            label=self.label,
                            geo_metadata=deepcopy(self.geo_metadata))
            
    def get_J(self,
              state,
              mask=None):
        """
        """    
        # get pixels indexes whose initial states are u
        # within the mask
        if mask is None:
            mask = np.ones_like(self)
        
        return np.where((self * mask).flat == int(state))[0]
    
    def get_V(self,
              J,
              final_states=None):
                
        V = self.flat[J]
                
        if final_states is not None:
            idx = np.isin(V, final_states)
            J = J[idx]
            V = V[idx]
            
        return(J, V)
    
    def get_X(self, 
              J,
              features):
        
        X = None
        
        for info in features:
            # switch according z_type
            if isinstance(info, Layer):
                # just get data
                x = info.flat[J]

            elif isinstance(info, int):
                # get distance data
                x = self.get_distance(state=info).flat[J]
                
            else:
                raise(TypeError('Unexpected feature info : ' + type(info) + '.'))

            # if X is not yet defined
            if X is None:
                X = x
            # else column stack
            else:
                X = np.column_stack((X, x))

        # if only one feature, reshape X as a column
        if len(X.shape) == 1:
            X = X[:, None]
        
        return(X)
    
    def get_distance(self, 
                     state, 
                     overwrite=False):
        if state not in self.distances.keys() or overwrite:
            v_matrix = (self == int(state)).astype(int)
            self.distances[state] = ndimage.distance_transform_edt(1 - v_matrix)
        
        return self.distances[state]
    
    def clean_distances(self):
        self.distances = {}
    
    def display(self,
                center,
                window,
                palette,
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
        
        ordered_palette = palette.sort(inplace=False)
        
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