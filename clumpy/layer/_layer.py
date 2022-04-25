# -*- coding: utf-8 -*-

import numpy as np
import os
from matplotlib import pyplot as plt
# from matplotlib import colors as mpl_colors
import rasterio

from ..tools._path import path_split, create_directories

import logging
logger = logging.getLogger('clumpy')

from ..tools._console import stop_log

class Layer(np.ndarray):
    """
    Layer base element
    """
    
    def __new__(cls, 
                input_array,
                label=None,
                band_tags=None,
                geo_metadata=None):
        obj = np.asarray(input_array).view(cls)
        
        obj.label = label
        obj.band_tags = band_tags
        obj.geo_metadata = geo_metadata
        
        return obj       

    def __str__(self):
        return(self.label)

    def get_band(self, band=0):
        return(self[band])
    
    def get_n_bands(self):
        return(self.shape[0])
    
    def save(self, path):
        folder_path, file_name, file_ext = path_split(path)
        create_directories(folder_path)
        
        data = np.array(self)
        
        # data shape preprocessing.
        # len(data.shape) should be equal to 3.
        if len(data.shape) == 1:
            data = data[None,None,:]
    
        elif len(data.shape) == 2:
            data = data[None, :, :]
        
        elif len(data.shape) > 4:
            logger.error("len(data.shape) is expected to be lower or equal to 3.")
            stop_log()
            raise(ValueError())
                
        # create parent directories if necessary
        create_directories(folder_path)

        # create file
        with rasterio.open(
            path,
            mode='w',
            driver=self.geo_metadata['driver'],
            height=data.shape[1],
            width=data.shape[2],
            count=data.shape[0],
            dtype=data.dtype,
            crs=self.geo_metadata['crs'],
            transform=self.geo_metadata['transform']
            ) as dst:
                dst.write(data)
                
                if self.band_tags is not None:
                    for band_i, tags in enumerate(self.band_tags):
                        dst.update_tags(band_i+1, **tags)

    def export(self, path, band=0, plane=False, rdc_only=False):
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
        
        if not rdc_only:
            os.system('rio convert '+self.path+' '+path+' --overwrite')
        
        data = self.get_band(band=band)
        
        if file_ext == 'rst':
            if not plane:
                rdc_file  = "file format : Idrisi Raster A.1\n"
                rdc_file += "file title  : \n"
                rdc_file += "data type   : byte\n"
                rdc_file += "file type   : binary\n"
                rdc_file += "columns     : "+str(data.shape[1])+"\n"
                rdc_file += "rows        : "+str(data.shape[0])+"\n"
                rdc_file += "ref.system  : spc83la3\n"
                rdc_file += "ref.units   : m\n"
                rdc_file += "unit dist.  : 1\n"
                rdc_file += "min.X       : "+str(self.geo_metadata['transform'][2])+"\n"
                rdc_file += "max.X       : "+str(self.geo_metadata['transform'][2] + self.geo_metadata['transform'][0] * self.get_data().shape[1])+"\n"
                rdc_file += "min.Y       : "+str(self.geo_metadata['transform'][5] + self.geo_metadata['transform'][4] * data.shape[0])+"\n"
                rdc_file += "max.Y       : "+str(self.geo_metadata['transform'][5])+"\n"
                rdc_file += "pos'n error : unspecified\n"
                rdc_file += "resolution  : "+str(np.abs(self.geo_metadata['transform'][0]))+"\n"
                rdc_file += "min.value   : "+str(data.min())+"\n"
                rdc_file += "max.value   : "+str(data.max())+"\n"
                rdc_file += "display min : "+str(data.min())+"\n"
                rdc_file += "display max : "+str(data.max())+"\n"
                rdc_file += "value units : unspecified\n"
                rdc_file += "value error : unspecified\n"
                rdc_file += "flag value  : none\n"
                rdc_file += "flag def 'n : none\n"
                rdc_file += "legend cats : 0\n"
                rdc_file += "lineage     : \n"
                rdc_file += "comment     :\n"
            
            else:
                rdc_file  = "file format : Idrisi Raster A.1\n"
                rdc_file += "file title  : \n"
                rdc_file += "data type   : byte\n"
                rdc_file += "file type   : binary\n"
                rdc_file += "columns     : "+str(data.shape[1])+"\n"
                rdc_file += "rows        : "+str(data.shape[0])+"\n"
                rdc_file += "ref.system  : plane\n"
                rdc_file += "ref.units   : m\n"
                rdc_file += "unit dist.  : 1.0\n"
                rdc_file += "min.X       : 0.0\n"
                rdc_file += "max.X       : "+str(data.shape[1])+"\n"
                rdc_file += "min.Y       : 0.0\n"
                rdc_file += "max.Y       : "+str(data.shape[0])+"\n"
                rdc_file += "pos'n error : unknown\n"
                rdc_file += "resolution  : 1.0\n"
                rdc_file += "min.value   : "+str(data.min())+"\n"
                rdc_file += "max.value   : "+str(data.max())+"\n"
                rdc_file += "display min : "+str(data.min())+"\n"
                rdc_file += "display max : "+str(data.max())+"\n"
                rdc_file += "value units : unspecified\n"
                rdc_file += "value error : unspecified\n"
                rdc_file += "flag value  : none\n"
                rdc_file += "flag def 'n : none\n"
                rdc_file += "legend cats : 0\n"
                rdc_file += "lineage     : \n"
                rdc_file += "comment     :\n"

            f = open(folder_path+'/'+file_name+'.rdc', "w")

            f.write(rdc_file)
            f.close()
    

    
    def export_asc(self, path, band=0, round=4):
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
        
        data = self.get_band(band=band)

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
                band=0,
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
        data = self.get_band(band=0)
        
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



def convert_raster_file(path_in, path_out):
    os.system('rio convert '+path_in+' '+path_out+' --overwrite')


