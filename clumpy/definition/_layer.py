# -*- coding: utf-8 -*-

from tifffile import TiffFile, imwrite # for tiff write and read

import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import re
# from sys import getsizeof
# from copy import deepcopy

# from ..utils import human_size

# from ..tools._validation import check_case

class _Layer:
    """Layer base element
    """

    def __init__(self,
                 name=None,
                 time=0,
                 path=None,
                 data=None,
                 copy_metadata=None):
        
        self.name = name
        self.time = time
        self.path = path
        self.data = data
        self.copy_metadata = copy_metadata
                
        # if path and data -> file creation
        if path is not None and data is not None:
            
            extratags = []
            
            # if copy metadata
            if copy_metadata is not None:
                
                # for each tag, append to extratags
                for _, tag in copy_metadata.tiff.pages[0].tags.items():
                    extratags.append((tag.code, tag.dtype, tag.count, tag.value, False))
            
            # create file
            imwrite(path, data=data, shape=data.shape, extratags=extratags)
            
        # read file
        self.tiff = TiffFile(path)
        
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
        
        data = self.tiff.asarray()
        
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
        
    # def get_size(self, print_value=True, human=True, return_size=False):
    #     """Get layer size

    #     Parameters
    #     ----------
    #     print_value : bool, default=``True``
    #         If ``True``, print the size.
    #     human : bool, default=``True``
    #         If ``True``, print the size in a readable way.
    #     return_value : bool, ``False``
    #         If ``True``, return the size.

    #     Returns
    #     -------
    #     (s): int
    #         The size in octet. Returned if ``return_size=True``.

    #     """
    #     s = getsizeof(self.data)
    #     if print_value:
    #         if human:
    #             sh = human_size(s)
    #             print(str(round(sh[0],2))+' '+sh[1])
    #         else:
    #             print(s)
    #     if return_size:
    #         return(s)
            
    # def copy(self):
    #     """Copy

    #     Returns
    #     -------
    #     None.

    #     """
    #     return(deepcopy(self))


class LandUseCoverLayer(_Layer):
    """Defines a Land Use Cover (LUC) layer.
    This layer can then used for the calibration stage or the allocation stage::
    
        luc1998 = clumpy.definition.layer.LayerLUC(name='LUC-1998',scale=15,time=0)
    
    Parameters
    ----------
    name : TYPE, optional
        DESCRIPTION. The default is None.
    time : TYPE, optional
        DESCRIPTION. The default is 0.
    path : TYPE, optional
        DESCRIPTION. The default is None.
    data : TYPE, optional
        DESCRIPTION. The default is None.
    copy_metadata : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    
    def __init__(self,
                 name=None,
                 time=0,
                 path=None,
                 data=None,
                 copy_metadata=None):
        
        super().__init__(name,
                         time,
                         path,
                         data,
                         copy_metadata)
    

    # def __init__(self,
    #              name=None,
    #              time=0,
    #              path=None,
    #              data=None,
    #              copy_metadata=None):
        
    #     super().__init__(name, time, path, data, copy_metadata)
        

    # def import_tiff(self, path, verbose=0):
    #     """Imports the layer data from a ``tif`` or a ``tiff`` file.
    #     The file extension can be something else than tiff. See `Image Pillow documentation, Image.open <https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.open>`_.
        
    #     Parameters
    #     ----------
    #     path : str
    #         path to the file.
    #     verbose : int, default=0
    #         level of verbosity.
    #     """
    #     if verbose >0:
    #         print("importing tiff file '" + path + "'")
        
    #     self._import_tiff_file(path)
        
    #     self.id_v, self.N_v = np.unique(self.data, return_counts=True)

    #     # self.get_dimensions()
    #     self.shape = np.shape(self.data)
    #     self.size = np.size(self.data)
        
    #     if verbose > 0:
    #         print("\t done, N_j=" + str(self.size))
        
    # def import_asc(self, path, verbose=0):
    #     """ imports the layer data from a ``asc`` file.

    #     Parameters
    #     ----------
    #     path : str
    #         path to the file.
    #     verbose : int, default=0
    #         level of verbosity.

    #     """
    #     f = open(path, 'r')
        
    #     lines = f.readlines()
        
    #     entete = lines[0:6]
    #     data = lines[6::]
        
    #     if verbose>0:
    #         print(entete)
    #         print(len(lines))
    #         print(len(data))
        
    #     infos =  {}
    #     for e in entete:
    #         chunks = re.split(' +', e)
    #         infos[chunks[0]] = float(chunks[1].replace('\n',''))
            
    #     data = [int(x.replace('\n','')) for x in data]
    #     data = np.array(data)
    #     self.data = data.reshape((int(infos['ncols']), int(infos['nrows'])))
        
    #     self.id_v, self.N_v = np.unique(self.data, return_counts=True)

    #     # self.get_dimensions()
    #     self.shape = np.shape(self.data)
    #     self.size = np.size(self.data)

    #     if verbose>0:
    #         print("\t done, N_j=" + str(self.size))

    # def __str__(self):
    #     txt = 'Object Map : ' + self.name + '\n'
    #     txt = txt + '(Nx, Ny) = ' + str(self.shape)+ '\n'
    #     txt = txt + 'mean(v) = ' + str(np.mean(self.data)) + '\n'
    #     unique, counts = np.unique(self.data, return_counts=True)
    #     for k in range(len(unique)):
    #         txt = txt + str(unique[k]) + ' ' + str(counts[k]) + '\n'
    #     txt = txt + ""
    #     return (txt)
    
    def set_style(self, style):
        """set display style

        Parameters
        ----------
        style : json
            The style. See example and user guide.

        """
        self.style_values = list(style.keys())
        self.style_names = [style[i][0] for i in style.keys()]
        self.style_colors = [style[i][1] for i in style.keys()]
    
    def display(self, center, window):
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
        values = self.style_values
        colors = self.style_colors
        names = self.style_names
        
        colors = colors[:-1] + [colors[-2]] + [colors[-1]]
        bounds = np.array(values+[values[-1]+1])-0.5
        
        cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        
        if type(window) == int:
            window = (window, window)
        
        x1 = int(center[0] - window[0]/2)
        x2 = int(center[0] + window[0]/2)
        y1 = int(center[1] - window[1]/2)
        y2 = int(center[1] + window[1]/2)
        
        if x1 < 0:
            x1 = 0
            x2 = window[0]
        if x2 >= self.data.shape[0]:
            x2 = int(self.data.shape[0])
            x1 = x2 - window[0]
        if y1 < 0:
            y1 = 0
            y2 = window[1]
        if y2 >= self.data.shape[1]:
            y2 = int(self.data.shape[1])
            y1 = y2 - window[1]
        
        plt.imshow(self.data[x1:x2, y1:y2], interpolation='none', cmap=cmap, norm=norm)
        plt.yticks([], [])
        plt.xticks([], [])
        cb = plt.colorbar()
        cb.set_ticks(values)
        cb.set_ticklabels(names)


class FeatureLayer(_Layer):
    """
    Defines an Explanatory Variable (EV) layer. This layer can then used for the calibration stage or the allocation stage::
    
        elevation = clumpy.definition.layer.LayerEV(name='elevation',time=0,scale=15)    
    
    Parameters
    ----------
    name : TYPE, optional
        DESCRIPTION. The default is None.
    time : TYPE, optional
        DESCRIPTION. The default is 0.
    path : TYPE, optional
        DESCRIPTION. The default is None.
    data : TYPE, optional
        DESCRIPTION. The default is None.
    copy_metadata : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """

    def __init__(self,
                 name=None,
                 time=0,
                 path=None,
                 data=None,
                 copy_metadata=None):
        
        super().__init__(name,
                         time,
                         path,
                         data,
                         copy_metadata)

    # def import_tiff(self, path):
    #     """
    #     Imports the layer data from a `tif` or a `tiff` file. The file extension can be something else than tiff. See `Image Pillow documentation, Image.open <https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.open>`_.
        
    #     Parameters
    #     ----------
    #     path : str
    #         path to the file.
    #     verbose : int, default=0
    #         level of verbosity.
    #     """
    #     print("importing tiff file '" + path + "'")
    #     self._import_tiff_file(path)
        
    #     # self.get_dimensions()
    #     self.shape = np.shape(self.data)
    #     self.size = np.size(self.data)

    #     print("\t done, N_j=" + str(self.size))

    # def __str__(self):
    #     txt = 'Object Map : ' + self.name + '\n'
    #     txt = txt + '(Nx, Ny) = ' + str(self.shape) + '\n'
    #     txt = txt + 'mean(data) = ' + str(np.mean(self.data)) + '\n'
    #     txt = txt + 'min(data) = ' + str(np.min(self.data)) + '\n'
    #     txt = txt + 'max(data) = ' + str(np.max(self.data)) + '\n'
    #     return (txt)


# class DistanceToVFeatureLayer(FeatureLayer):
#     """
#     Defines a distance to a state as a layer. This layer can then used for the calibration stage or the allocation stage::
    
#         distance_to_2 = clumpy.definition.layer.LayerEV(name='elevation',time=0,scale=15)    
    
#     However, it is recommended to prefer the case's method clumpy.definition.Case.add_distance_to_v_as_feature.
    
#     Parameters
#     ----------
#     v : int
#         The state to compute the distance from.
    
#     layer_LUC : LandUseCoverLayer
#         The land use cover used to compute the distance.
        
#     name : string (default=``None``)
#         The layer name. If none, name is defined as ``'distance_to_v_'+str(v)``.
#     """
    
#     def __init__(self, v, layer_LUC, name=None):
        
#         if type(name)==type(None):
#             name = 'distance_to_v_'+str(v)
        
#         super().__init__(name=name, scale=layer_LUC.scale)
#         self.v = v
#         self.layer_LUC = layer_LUC
#         self.update(layer_LUC=self.layer_LUC)
#         layer_LUC.distance2v.append(self)

#     def update(self, layer_LUC):
#         v_matrix = (layer_LUC.data == self.v).astype(int)
#         self.data = ndimage.distance_transform_edt(1 - v_matrix) * layer_LUC.scale


# class TransitionProbabilityLayers():
#     """
#     Defines a set of :math:`P(vf|vi,z)` layers.
    
#     Notes
#     -----
#         The set of layers is defined as a dictionary ``self.layers`` whose keys are :math:`(v_i,v_f)`.
#     """
    
#     def __init__(self):
#         self.layers = {}
    
#     def add_layer(self, vi, vf, data=None, path=None):
#         """
#         Add a layer to the set.        

#         Parameters
#         ----------
#         vi : int
#             initial state.
#         vf : int
#             final state.
#         data : numpy array (default=None)
#             The :math:`P(vf|vi,z)` data layer. If None, ``path`` is expected.
#         path : string (default=None)
#             The path to :math:`P(vf|vi,z)` tiff file. If None, ``data`` is expected.

#         """
#         self.layers[(vi, vf)] = _TransitionProbabilityLayer(vi, vf, data, path)
    
#     def copy(self):
#         """
#         Make a copy.

#         Returns
#         -------
#         c : TransitionProbabilityLayers
#             A copy of the current object.

#         """
#         return(deepcopy(self))
        
#     def export_all(self, path:str):
#         """
#         Export all layers as tif files through a zip archive file.

#         Parameters
#         ----------
#         path : str
#             Output zip file path. Required new folders are created without error raising.
            
#         Notes
#         -----
#         All layer files inside the zip archive file are named as following : ``'P_vf' + str(vf) + '__vi' + str(vi) + '_z.tif'``.
#         """
#         files_names = []
#         # folder_name = os.path.dirname(path)
        
#         for layer in self.layers.values():
#             files_names.append(layer.name+'.tif')
#             layer.export_tiff(files_names[-1])
        
#         command = 'zip .temp_P_vf__vi_z.zip'
#         for file_name in files_names:
#             command += ' '+file_name
#         os.system(command)
        
#         command = 'rm '
#         for file_name in files_names:
#             command += ' '+file_name
#         os.system(command)
        
#         # create folder if not exists
#         folder_name = os.path.dirname(path)
#         if not os.path.exists(folder_name) and folder_name!= '':
#             os.makedirs(folder_name)
        
#         os.system('mv .temp_P_vf__vi_z.zip '+path)
        
#     def import_all(self, path:str):
#         """
#         Import all layers from a zip archive file.

#         Parameters
#         ----------
#         path : str
#             The zip file to import.
            
#         Notes
#         -----
#         All layer files inside the zip archive file are expected to be named as following : ``'P_vf' + str(vf) + '__vi' + str(vi) + '_z.tif'``.

#         """
#         os.system('unzip '+path+' -d '+path+'.out')
        
#         files = os.listdir(path+'.out/')
        
#         self.layers = {}
#         for file in files:
#             start = 'vf'
#             end = '__'
#             vf = file[file.find(start)+len(start):file.rfind(end)]
            
#             start = 'vi'
#             end = '_z'
#             vi = file[file.find(start)+len(start):file.rfind(end)]
            
            
#             self.add_layer(vi=int(vi), vf=int(vf), path=path+'.out/'+file)
            
#         os.system('rm -R '+path+'.out')
            

# class _TransitionProbabilityLayer(_Layer):
#     """
#     Defines a P(vf|vi,z) layer. This layer can then used for the allocation stage.
#     """

#     def __init__(self, vi, vf, data=None, path=None):
#         super().__init__('P_vf' + str(vf) + '__vi' + str(vi) + '_z')
#         self.vi = vi
#         self.vf = vf
#         # self.Tif.layer_P_vf__vi_z = self

#         if type(data) == np.ndarray:
#             self.data = data

#         if type(path) != type(None):
#             self.import_tiff(path)

#     def import_tiff(self, path):
#         """
#         Imports the layer data from a `tif` or a `tiff` file. The file extension can be something else than tiff. See `Image Pillow documentation, Image.open <https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.open>`_.
        
#         :param path: path to the file
#         :type path: str
#         """
#         print("importing tiff file '" + path + "'")
#         self.path = path

#         img = Image.open(path)  # image loading
#         img.load()
#         dtype = "float"
#         self.data = np.asarray(img, dtype=dtype)  # conversion en numpy

#         # self.get_dimensions()
#         self.shape = np.shape(self.data)
#         self.size = np.size(self.data)

#         print("\t done, N_j=" + str(self.size))


