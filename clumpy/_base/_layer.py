# -*- coding: utf-8 -*-



import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
import rasterio

from ..tools._data import ndarray_suitable_integer_type
from ..tools._path import path_split, create_directories

class Layer:
    """Layer base element
    """

    def __init__(self,
                 label=None,
                 time=0,
                 path=None,
                 data=None,
                 copy_geo=None):
        
        self.label = label
        self.time = time
        self.path = path
        self.copy_geo = copy_geo
                
        # if path and data -> file creation
        if path is not None and data is not None:
            
            data = data.copy()
            
            # data shape preprocessing.
            # len(data.shape) should be equal to 3.
            if len(data.shape) == 1:
                data = data[None,None,:]
        
            elif len(data.shape) == 2:
                data = data[None, :, :]
            
            elif len(data.shape) > 3:
                raise(ValueError("len(data.shape) is expected to be lower or equal to 3."))

            # choose an appropriate dtype.
            data = ndarray_suitable_integer_type(data)

            driver = None
            crs = None
            transform = None
            
            # if copy metadata
            if self.copy_geo is not None:
                driver = self.copy_geo.raster_.driver
                crs = self.copy_geo.raster_.crs
                transform = self.copy_geo.raster_.transform
            
            # create parent directories if necessary
            folder_path, file_name, file_ext = path_split(path)
            create_directories(folder_path)
            
            # create file
            with rasterio.open(
                self.path,
                mode='w',
                driver=driver,
                height=data.shape[1],
                width=data.shape[2],
                count=data.shape[0],
                dtype=data.dtype,
                crs=crs,
                transform=transform
                ) as dst:
                    dst.write(data)
        
        # read file
        self.raster_ = rasterio.open(self.path)

    def get_data(self, band=1):
        return(self.raster_.read(band))

    def __repr__(self):
        return(self.label)
        
    def export_asc(self, path, verbose=0):
        """Export the layer data as an ``asc`` file in order to use it through CLUES and CLUMondo.
        
        Parameters
        ----------
        path : str
            path to the file.
        verbose : int, default=0
            level of verbosity.
        """
        if verbose>0:
            print("[" + self.label + "] exporting tiff file in " + path + "...")
        
         # create folder if not exists
        folder_label = os.path.dirlabel(path)
        if not os.path.exists(folder_label) and folder_label!= '':
            os.makedirs(folder_label)
        
        data = self.raster_.read(1)
        
        np.savetxt(path, data.astype(int), delimiter=' ', fmt='%i')
        
        f= open(path,"r")
        
        data = f.read()
        
        entete =  "ncols        "+str(data.shape[0])+"\n"
        entete += "nrows        "+str(data.shape[1])+"\n"
        entete += "xllcorner    0.0\n"
        entete += "yllcorner    -"+str(float(data.shape[1]))+"\n"
        entete += "cellsize     1.0\n"
        
        f= open(path,"w")
        
        f.write(entete+data)
        f.close()
        
        if verbose>0:
            print('done')
            
    def display(self,
                center,
                window,
                show=True,
                colorbar = True,
                **kwargs_imshow):
        """
        Display the layer.
        
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
        **kwargs_imshow : kwargs
            Keyword arguments passed to the ``plt.imshow()`` function.

        Returns
        -------
        plt : matplotlib.pyplot
            Pyplot object

        """
        
        if type(center) is int:
            center = np.unravel_index(center, self.get_data().shape)
        
        data = self.get_data()
        
        if type(window) == int:
            window = (window, window)
        
        x1 = int(center[0] - window[0]/2)
        x2 = int(center[0] + window[0]/2)
        y1 = int(center[1] - window[1]/2)
        y2 = int(center[1] + window[1]/2)
        
        if x1 < 0:
            x1 = 0
            x2 = window[0]
        if x2 >= data.shape[0]:
            x2 = int(data.shape[0])
            x1 = x2 - window[0]
        if y1 < 0:
            y1 = 0
            y2 = window[1]
        if y2 >= data.shape[1]:
            y2 = int(data.shape[1])
            y1 = y2 - window[1]
            
        plt.imshow(data[x1:x2, y1:y2], **kwargs_imshow)
        plt.yticks([], [])
        plt.xticks([], [])
        
        if colorbar:
            plt.colorbar()
        if show:
            plt.show()
        return(plt)

class LandUseLayer(Layer):
    """Define a Land Use Cover (LUC) layer.
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
    copy_geo : :class:`LandUseLayer`, default=None
        The layer from whose geo metadata are copied.
        If ``None``, geo metadata are set to ``None``.
    palette : Palette
        The states palette.

    Attributes
    ----------
    raster_ : :class:`rasterio.io.DatasetReader`
        The unbuffered data and metadata reader object, provided by :mod:`rasterio`.
    style_ : list of dict
        The style used for displaying.
    """
    
    def __init__(self,
                 label=None,
                 time=0,
                 path=None,
                 data=None,
                 copy_geo=None,
                 palette=None):
        
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

        if ~np.all(np.equal(np.unique(self.get_data()), np.array([0,1]))):
            raise(ValueError("Unexpected mask layer. Mask layer should be only composed by '0' and '1' values."))

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
    
    low_bound : None or float, default=None
        If float, a low bound is set. Used in density estimation methods as GKDE.
    
    high_bound : None or float, default=None
        If float, a high bound is set. Used in density estimation methods as GKDE.
    
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
                 low_bound = None,
                 high_bound = None,
                 copy_geo=None):
        
        super().__init__(label=label,
                         time=time,
                         path=path,
                         data=data,
                         copy_geo=copy_geo)
        
        self.low_bound = low_bound
        self.high_bound = high_bound

