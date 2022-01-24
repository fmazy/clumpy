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

        folder_path, file_name, file_ext = path_split(path)

        if self.label is None:
            self.label = file_name

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

            driver = None
            crs = None
            transform = None
            
            # if copy metadata
            if self.copy_geo is not None:
                driver = self.copy_geo.raster_.driver
                crs = self.copy_geo.raster_.crs
                transform = self.copy_geo.raster_.transform
            
            # create parent directories if necessary
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

    def export(self, path):
        """Export the layer according to the file extension. See GDAL for available extenstions.
        For floating rst, the data should be np.float32.
        Parameters
        ----------
        path : str
            path to the file.
        """
        # create folder if not exists
        folder_path, file_name, file_ext = path_split(path)
        create_directories(folder_path)

        os.system('rio convert '+self.path+' '+path+' --overwrite')

        if file_ext == 'rst':
            rdc_file = "file format: Idrisi Raster A.1\n"
            rdc_file += "file title: \n"
            rdc_file += "data type: byte\n"
            rdc_file += "file type: binary\n"
            rdc_file += "columns: "+str(self.get_data().shape[1])+"\n"
            rdc_file += "rows: "+str(self.get_data().shape[0])+"\n"
            rdc_file += "ref.system: spc83la3\n"
            rdc_file += "ref.units: m\n"
            rdc_file += "unit dist.: 1\n"
            rdc_file += "min.X: "+str(self.raster_.transform[2])+"\n"
            rdc_file += "max.X: "+str(self.raster_.transform[2] + self.raster_.transform[0] * self.get_data().shape[1])+"\n"
            rdc_file += "min.Y: "+str(self.raster_.transform[5] + self.raster_.transform[4] * self.get_data().shape[0])+"\n"
            rdc_file += "max.Y: "+str(self.raster_.transform[5])+"\n"
            rdc_file += "pos'n error : unspecified\n"
            rdc_file += "resolution: "+str(np.abs(self.raster_.transform[0]))+"\n"
            rdc_file += "min.value: "+str(self.get_data().min())+"\n"
            rdc_file += "max.value: "+str(self.get_data().max())+"\n"
            rdc_file += "display min: "+str(self.get_data().min())+"\n"
            rdc_file += "display max: "+str(self.get_data().max())+"\n"
            rdc_file += "value units: unspecified\n"
            rdc_file += "value error: unspecified\n"
            rdc_file += "flag value: none\n"
            rdc_file += "flag def 'n  : none\n"
            rdc_file += "legend cats: 0\n"
            rdc_file += "lineage: \n"
            rdc_file += "comment:\n"

            f = open(folder_path+'/'+file_name+'.rdc', "w")

            f.write(rdc_file)
            f.close()

    def export_asc(self, path, round=4):
        """Export the layer data as an ``asc`` file in order to use it through CLUES and CLUMondo.
        
        Parameters
        ----------
        path : str
            path to the file.
        round : int, default=4
            Number of decimals to keep if data is not an dtype integer array.
        """
        
         # create folder if not exists
        folder_path, file_name, file_ext = path_split(path)
        create_directories(folder_path)
        
        data = self.get_data()

        if np.issubdtype(data.dtype, np.integer):
            fmt = '%i'
        else:
            fmt = '%.'+str(round)+'f'

        np.savetxt(path, data, delimiter=' ', fmt=fmt)
        
        f= open(path,"r")
        
        text_data = f.read()

        entete =  "ncols        "+str(data.shape[0])+"\n"
        entete += "nrows        "+str(data.shape[1])+"\n"
        entete += "xllcorner    0.0\n"
        entete += "yllcorner    -"+str(float(data.shape[1]))+"\n"
        entete += "cellsize     1.0\n"
        
        f= open(path,"w")
        
        f.write(entete+text_data)
        f.close()
            
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
        
        if type(center) is int or type(center) == np.int64:
            center = np.unravel_index(center, self.get_data().shape)
            print('c', center)
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
                 bounded = None,
                 copy_geo=None):
        
        super().__init__(label=label,
                         time=time,
                         path=path,
                         data=data,
                         copy_geo=copy_geo)
        
        self.bounded = bounded

def convert_raster_file(path_in, path_out):
    os.system('rio convert '+path_in+' '+path_out+' --overwrite')

layers = {'land_use' : LandUseLayer,
          'feature' : FeatureLayer,
          'mask' : MaskLayer}
