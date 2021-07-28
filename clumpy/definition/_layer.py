# -*- coding: utf-8 -*-



import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
import rasterio

class _Layer:
    """Layer base element
    """

    def __init__(self,
                 name=None,
                 time=0,
                 path=None,
                 data=None,
                 copy_geo=None):
        
        self.name = name
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
            
            driver = None
            crs = None
            transform = None
            
            # if copy metadata
            if self.copy_geo is not None:
                driver = self.copy_geo.raster_.driver
                crs = self.copy_geo.raster_.crs
                transform = self.copy_geo.raster_.transform
            
            
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
        return('Layer()')
        
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
            print("[" + self.name + "] exporting tiff file in " + path + "...")
        
         # create folder if not exists
        folder_name = os.path.dirname(path)
        if not os.path.exists(folder_name) and folder_name!= '':
            os.makedirs(folder_name)
        
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



class LandUseCoverLayer(_Layer):
    """Defines a Land Use Cover (LUC) layer.
    This layer can then used for the calibration stage or the allocation stage.
    
    Parameters
    ----------
    name : str, default=``None``
        The layer name.
    time : float, default=``0``
        The layer time (year unit).
    path : str
        The tiff file path.
        If ``data`` is provided, a new tiff file will be created according to this path
        and this operation overwrites the file path if exists.
    data : :class:`numpy.ndarray`, defaul=None
        The data to write. If ``None``, no writing is made.
    copy_geo : :class:`LandUseCoverLayer`, default=None
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
                 name=None,
                 time=0,
                 path=None,
                 data=None,
                 copy_geo=None):
        
        super().__init__(name=name,
                         time=time,
                         path=path,
                         data=data,
                         copy_geo=copy_geo)
    
        self.style_ = {}

    def __repr__(self):
        return('LandUseCoverLayer()')
        
    def import_style(self,
                      path):
        """
        Import legend through qml file provided by QGis. Alpha is not supported.

        Parameters
        ----------
        path : str
            The qml file path

        """
        from xml.dom import minidom

        # parse an xml file by name
        mydoc = minidom.parse(path)
        items = mydoc.getElementsByTagName('paletteEntry')
        
        self.style_ = []
        
        for elem in items:
            
            self.style_.append({'color':elem.attributes['color'].value,
                                'label':elem.attributes['label'].value,
                                'value':int(elem.attributes['value'].value)})
    
    def clean_style(self, null='#ffffff'):
        """
        Clean the style_ attr

        Parameters
        ----------
        null : TYPE, optional
            DESCRIPTION. The default is '#ffffff'.

        Returns
        -------
        None.

        """
        style = []
        
        u_unique = list(np.unique(self.raster_.read(1)))
        
        for s in self.style_:
            if s['value'] in u_unique:
                style.append(s)
                u_unique.pop(u_unique.index(s['value']))
        
        if len(u_unique) > 0:
            for u in u_unique:
                style.append({'color':'#ffffff',
                              'label':'null',
                              'value':u})
        
        self.style_ = style
    
    def display(self, center, window, show=True, colorbar=True):
        """Display the land use cover layer through python console with matplotlib.

        Parameters
        ----------
        values : list of int
            List of displayed states.
        colors : list of str
            List of colors in HTML format.
        names : list of names
            List of state names.
        center : tuple of two integers
            Center position as a tuple.
        window : tuple of two integers
            Window dimensions as a tuple.
        """
        values = np.array([s['value'] for s in self.style_])
        colors = [s['color'] for s in self.style_]
        labels = [s['label'] for s in self.style_]
        
        index_sorted = list(np.argsort(values))
        
        values = [values[i] for i in index_sorted]
        colors = [colors[i] for i in index_sorted]
        labels = [labels[i] for i in index_sorted]
        
        data = self.raster_.read(1)
        
        colors = colors[:-1] + [colors[-2]] + [colors[-1]]
        bounds = np.array(values+[values[-1]+1])-0.5
        
        cmap = mpl_colors.ListedColormap(colors)
        norm = mpl_colors.BoundaryNorm(bounds, cmap.N)
        
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

        plt.imshow(data[x1:x2, y1:y2], interpolation='none', cmap=cmap, norm=norm)
        plt.yticks([], [])
        plt.xticks([], [])

        if colorbar:
            cb = plt.colorbar()
            cb.set_ticks(values)
            cb.set_ticklabels(labels)

        if show:
           plt.show()
        return(plt)


class FeatureLayer(_Layer):
    """Defines a feature layer.
    This layer can then used for the calibration stage or the allocation stage.
    
    Parameters
    ----------
    name : str, default=``None``
        The layer name.
    time : float, default=``0``
        The layer time (year unit).
    path : str
        The tiff file path.
        If ``data`` is provided, a new tiff file will be created according to this path
        and this operation overwrites the file path if exists.
    data : :class:`numpy.ndarray`, defaul=None
        The data to write. If ``None``, no writing is made.
    copy_geo : :class:`LandUseCoverLayer`, default=None
        The layer from whose geo metadata are copied.
        If ``None``, geo metadata are set to ``None``.

    Attributes
    ----------
    raster_ : :class:`rasterio.io.DatasetReader`
        The unbuffered data and metadata reader object, provided by :mod:`rasterio`.
    """

    def __init__(self,
                 name=None,
                 time=0,
                 path=None,
                 data=None,
                 low_bounded=False,
                 high_bounded=False,
                 copy_geo=None):
        
        super().__init__(name=name,
                         time=time,
                         path=path,
                         data=data,
                         copy_geo=copy_geo)
        
        self.low_bounded = low_bounded
        self.high_bounded = high_bounded

    def __repr__(self):
        return('FeatureLayer()')
