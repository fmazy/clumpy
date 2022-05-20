# -*- coding: utf-8 -*-

import numpy as np
import os
from matplotlib import pyplot as plt
# from matplotlib import colors as mpl_colors
import rasterio
from copy import deepcopy

from ..tools._path import path_split, create_directories

import logging
logger = logging.getLogger('clumpy')

from ..tools._console import stop_log

structures = {
    'queen' : np.ones((3, 3)),
    'rook' : np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])
}

class Layer(np.ndarray):
    """
    Layer base element
    """
    
    def __new__(cls, 
                input_array,
                label=None,
                dtype=None,
                geo_metadata=None):
        
        obj = np.asarray(input_array, dtype=dtype).view(cls)
        
        obj.label = label
        obj.geo_metadata = geo_metadata
        
        if obj.geo_metadata is None:
            obj.geo_metadata = {'driver': 'GTiff',
             'crs': None,
             'transform': rasterio.Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)}
        
        return obj      
    
    def copy(self):
        return Layer(self,
                     label=self.label,
                     geo_metadata=deepcopy(self.geo_metadata))

    def __str__(self):
        return(self.label)
    
    def unravel_index(self, j):
        return np.unravel_index(j, self.shape)
    
    def ravel_index(self, x, y):
        return np.ravel_multi_index([x, y], self.shape)
    
    def save(self, path):
        self._save(path=path)

    def _save(self, path, band_tags=None):
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
                
                if band_tags is not None:
                    for band_i, tags in enumerate(band_tags):
                        dst.update_tags(band_i+1, **tags)

    def export(self, path, plane=False, rdc_only=False):
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
        
        
        if file_ext == 'rst':
            if not plane:
                rdc_file  = "file format : Idrisi Raster A.1\n"
                rdc_file += "file title  : \n"
                rdc_file += "data type   : byte\n"
                rdc_file += "file type   : binary\n"
                rdc_file += "columns     : "+str(self.shape[1])+"\n"
                rdc_file += "rows        : "+str(self.shape[0])+"\n"
                rdc_file += "ref.system  : spc83la3\n"
                rdc_file += "ref.units   : m\n"
                rdc_file += "unit dist.  : 1\n"
                rdc_file += "min.X       : "+str(self.geo_metadata['transform'][2])+"\n"
                rdc_file += "max.X       : "+str(self.geo_metadata['transform'][2] + self.geo_metadata['transform'][0] * self.shape[1])+"\n"
                rdc_file += "min.Y       : "+str(self.geo_metadata['transform'][5] + self.geo_metadata['transform'][4] * self.shape[0])+"\n"
                rdc_file += "max.Y       : "+str(self.geo_metadata['transform'][5])+"\n"
                rdc_file += "pos'n error : unspecified\n"
                rdc_file += "resolution  : "+str(np.abs(self.geo_metadata['transform'][0]))+"\n"
                rdc_file += "min.value   : "+str(self.min())+"\n"
                rdc_file += "max.value   : "+str(self.max())+"\n"
                rdc_file += "display min : "+str(self.min())+"\n"
                rdc_file += "display max : "+str(self.max())+"\n"
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
                rdc_file += "columns     : "+str(self.shape[1])+"\n"
                rdc_file += "rows        : "+str(self.shape[0])+"\n"
                rdc_file += "ref.system  : plane\n"
                rdc_file += "ref.units   : m\n"
                rdc_file += "unit dist.  : 1.0\n"
                rdc_file += "min.X       : 0.0\n"
                rdc_file += "max.X       : "+str(self.shape[1])+"\n"
                rdc_file += "min.Y       : 0.0\n"
                rdc_file += "max.Y       : "+str(self.shape[0])+"\n"
                rdc_file += "pos'n error : unknown\n"
                rdc_file += "resolution  : 1.0\n"
                rdc_file += "min.value   : "+str(self.min())+"\n"
                rdc_file += "max.value   : "+str(self.max())+"\n"
                rdc_file += "display min : "+str(self.min())+"\n"
                rdc_file += "display max : "+str(self.max())+"\n"
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
            center = self.unravel_index(center)
            print('c', center)
        
        if type(window) == int:
            window = (window, window)
        
        x1 = int(center[0] - window[0]/2)
        x2 = int(center[0] + window[0]/2)
        y1 = int(center[1] - window[1]/2)
        y2 = int(center[1] + window[1]/2)
        
        if x1 < 0:
            x1 = 0
            x2 = window[0]
        if x2 >= self.shape[0]:
            x2 = int(self.shape[0])
            x1 = x2 - window[0]
        if y1 < 0:
            y1 = 0
            y2 = window[1]
        if y2 >= self.shape[1]:
            y2 = int(self.shape[1])
            y1 = y2 - window[1]
            
        plt.imshow(self[x1:x2, y1:y2], **kwargs_imshow)
        plt.yticks([], [])
        plt.xticks([], [])
        
        if colorbar:
            plt.colorbar()
        if show:
            plt.show()
        return(plt)
    
    def get_neighbors_id(self, 
                         j, 
                         neighbors_structure='rook'):
        shape = self.shape
        
        if neighbors_structure == 'queen':
            j_neighbors = j + np.array([- shape[1],     # 0, top
                                        - shape[1] + 1, # 1, top-right
                                          1,            # 2, right
                                          shape[1] + 1, # 3, bottom-right
                                          shape[1],     # 4, bottom
                                          shape[1] - 1, # 5, bottom-left
                                        - 1,            # 6, left
                                        - shape[1] - 1])# 7, top-left
            
            # remove if side pixel
            id_to_remove = []
            if (j + 1) % shape[1] == 0: # right side
                id_to_remove += [1,2,3]
            if j % shape[1] == 0: # left side
                id_to_remove += [5,6,7]
            if j >= shape[0]*shape[1] - shape[1]: # bottom side
                id_to_remove += [3,4,5]
            if j < shape[1]: # top side
                id_to_remove += [0,1,7]
            
            j_neighbors = np.delete(j_neighbors, id_to_remove)
            
        elif neighbors_structure == 'rook':
            j_neighbors = j + np.array([- shape[1],     # 0, top
                                          1,            # 1, right
                                          shape[1],     # 2, bottom
                                        - 1])           # 3, left
            
            # remove if side pixel
            id_to_remove = []
            if (j + 1) % shape[1] == 0: # right side
                id_to_remove += [1]
            if j % shape[1] == 0: # left side
                id_to_remove += [3]
            if j >= shape[0]*shape[1] - shape[1]: # bottom side
                id_to_remove += [2]
            if j < shape[1]: # top side
                id_to_remove += [0]
            
            j_neighbors = np.delete(j_neighbors, id_to_remove)
            
        else:
            print('ERROR, unexpected neighbors_structure')
        
        return(j_neighbors)


def convert_raster_file(path_in, path_out):
    os.system('rio convert '+path_in+' '+path_out+' --overwrite')


