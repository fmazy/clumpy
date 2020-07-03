"""
The definition layer of maps
"""

from PIL import Image  # it seems to be a bit limitation of this library
# from libtiff import TIFF # this one seems to be extended. see https://stackoverflow.com/questions/7569553/working-with-tiffs-import-export-in-python-using-numpy
import numpy as np
from scipy import ndimage
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import re
from copy import deepcopy


class _Layer:
    """
    Defines a layer. It is expected to define a layer according to its type by the child classes :
        
        * :class:`.LayerLUC`
        * :class:`.LayerEV`
    """

    def __init__(self, name=None, scale=1):
        self.name = name
        self.scale = scale

    # def get_dimensions(self):
    #     """
    #     Computes, saves and returns the layer dimension.
        
    #     :Returns: a int tuple of dimensions ``self.Nx, self.Ny``
    #     """
    #     self.Nx = len(self.data[:, 0])
    #     self.Ny = len(self.data[0, :])
    #     self.Nj = self.Nx * self.Ny

    #     return (self.Nx, self.Ny)

    def import_numpy(self, data, sound=1):
        """
        Set the layer data by importing a numpy matrix.
        
        :param data: the data
        :type data: numpy array
        """
        if sound>0:
            print("importing numpy data")
        self.data = data

        # self.get_dimensions()
        self.size = np.size(self.data)
        
        if sound>0:
            print("\t done, N_j=" + str(self.size))

    def export_tiff(self, path, mode=None):
        """
        Export the layer data as a ``tif`` or a ``tiff`` file. The file extension can be something else than tiff. See `Image Pillow documentation, Image.save <https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.save>`_.
        
        using mode="I" is required by Dinamica for LUC maps.
        
        :param path: output file path
        :type path: str
        """
        print("[" + self.name + "] exporting tiff file in " + path + "...")
        img = Image.fromarray(self.data, mode=mode)
        
        # create folder if not exists
        folder_name = os.path.dirname(path)
        if not os.path.exists(folder_name) and folder_name!= '':
            os.makedirs(folder_name)
            
        img.save(path)
        
    def export_asc(self, path):
        """
        Export the layer data as an ``asc`` file in order to use it through CLUES and CLUMondo.
        """
        print("[" + self.name + "] exporting tiff file in " + path + "...")
        
         # create folder if not exists
        folder_name = os.path.dirname(path)
        if not os.path.exists(folder_name) and folder_name!= '':
            os.makedirs(folder_name)
        
        np.savetxt(path, self.data.astype(int), delimiter=' ', fmt='%i')
        
        f= open(path,"r")
        
        data = f.read()
        
        entete =  "ncols        "+str(self.data.shape[0])+"\n"
        entete += "nrows        "+str(self.data.shape[1])+"\n"
        entete += "xllcorner    0.0\n"
        entete += "yllcorner    -"+str(float(self.data.shape[1]))+"\n"
        entete += "cellsize     1.0\n"
        
        f= open(path,"w")
        
        f.write(entete+data)
        f.close()
        
        print('done')


class LandUseCoverLayer(_Layer):
    """
    Defines a Land Use Cover (LUC) layer. This layer can then used for the calibration stage or the allocation stage::
    
        luc1998 = clumpy.definition.layer.LayerLUC(name='LUC-1998',scale=15,time=0)
    
    :param name: the map name
    :type name: string, optional
    :param scale: the pixel side real length in meters
    :type scale: float
    :param time: the map time
    :type time: time
    """

    def __init__(self, name=None, scale=1, time=0, path=None, sound=1):
        super().__init__(name, scale)
        self.time = time

        self.distance2v = []
        self.Z = []

        if path != None:
            self.import_tiff(path, sound)

    def import_tiff(self, path, sound=1):
        """
        Imports the layer data from a ``tif`` or a ``tiff`` file. The file extension can be something else than tiff. See `Image Pillow documentation, Image.open <https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.open>`_.
        
        :param path: path to the file
        :type path: str
        """
        if sound >0:
            print("importing tiff file '" + path + "'")
        self.path = path

        img = Image.open(path)  # image loading
        img.load()
        dtype = "int32"  # "float" for EF
        self.data = np.asarray(img, dtype=dtype)  # conversion en numpy

        self.id_v, self.N_v = np.unique(self.data, return_counts=True)

        # self.get_dimensions()
        self.shape = np.shape(self.data)
        self.size = np.size(self.data)
        
        if sound > 0:
            print("\t done, N_j=" + str(self.size))
        
    def import_asc(self, path):
        f = open(path, 'r')
        
        lines = f.readlines()
        
        entete = lines[0:6]
        data = lines[6::]
        
        print(entete)
        print(len(lines))
        print(len(data))
        
        infos =  {}
        for e in entete:
            chunks = re.split(' +', e)
            infos[chunks[0]] = float(chunks[1].replace('\n',''))
            
        data = [int(x.replace('\n','')) for x in data]
        data = np.array(data)
        self.data = data.reshape((int(infos['ncols']), int(infos['nrows'])))
        
        self.id_v, self.N_v = np.unique(self.data, return_counts=True)

        # self.get_dimensions()
        self.shape = np.shape(self.data)
        self.size = np.size(self.data)

        print("\t done, N_j=" + str(self.size))

    def __str__(self):
        txt = 'Object Map : ' + self.name + '\n'
        txt = txt + '(Nx, Ny) = ' + str(self.shape)+ '\n'
        txt = txt + 'mean(v) = ' + str(np.mean(self.data)) + '\n'
        unique, counts = np.unique(self.data, return_counts=True)
        for k in range(len(unique)):
            txt = txt + str(unique[k]) + ' ' + str(counts[k]) + '\n'
        txt = txt + ""
        return (txt)
    
    def display(self, values, colors, names, center, window):
        """
        Display the land use cover layer through python console with matplotlib.

        Parameters
        ----------
        values : [int]
            List of displayed states.
        colors : [str]
            List of colors in HTML format.
        names : [str]
            List of state names.
        center : (int,int)
            Center position as a tuple.
        window : (int,int)
            Window dimensions as a tuple.
        """
        colors = colors[:-1] + [colors[-2]] + [colors[-1]]
        bounds = np.array(values+[values[-1]+1])-0.5
        
        cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                
        x1 = int(center[0] - window/2)
        x2 = int(center[0] + window/2)
        y1 = int(center[1] - window/2)
        y2 = int(center[1] + window/2)
        
        if x1 < 0:
            x1 = 0
            x2 = window
        if x2 >= self.data.shape[0]:
            x2 = int(self.data.shape[0])
            x1 = x2 - window
        if y1 < 0:
            y1 = 0
            y2 = window
        if y2 >= self.data.shape[1]:
            y2 = int(self.data.shape[1])
            y1 = y2 - window
        
        plt.imshow(self.data[x1:x2, y1:y2], interpolation='none', cmap=cmap, norm=norm)
        plt.yticks([], [])
        plt.xticks([], [])
        cb = plt.colorbar()
        cb.set_ticks([1,2,3])
        cb.set_ticklabels(names)
        cb.set_label('$v_f$', fontsize=14)


class FeatureLayer(_Layer):
    """
    Defines an Explanatory Variable (EV) layer. This layer can then used for the calibration stage or the allocation stage::
    
        elevation = clumpy.definition.layer.LayerEV(name='elevation',time=0,scale=15)    
    
    :param name: the map name
    :type name: string, optional
    :param scale: the pixel side real length in meters
    :type scale: float
    :param time: the map time
    :type time: time
    """

    def __init__(self, name=None, scale=1, time=0, path=None):
        super().__init__(name, scale)
        self.time = time

        if path != None:
            self.import_tiff(path)

    def import_tiff(self, path):
        """
        Imports the layer data from a `tif` or a `tiff` file. The file extension can be something else than tiff. See `Image Pillow documentation, Image.open <https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.open>`_.
        
        :param path: path to the file
        :type path: str
        """
        print("importing tiff file '" + path + "'")
        self.path = path

        img = Image.open(path)  # image loading
        img.load()
        dtype = "float"
        self.data = np.asarray(img, dtype=dtype)  # conversion en numpy

        # self.get_dimensions()
        self.shape = np.shape(self.data)
        self.size = np.size(self.data)

        print("\t done, N_j=" + str(self.size))

    def __str__(self):
        txt = 'Object Map : ' + self.name + '\n'
        txt = txt + '(Nx, Ny) = ' + str(self.shape) + '\n'
        txt = txt + 'mean(data) = ' + str(np.mean(self.data)) + '\n'
        txt = txt + 'min(data) = ' + str(np.min(self.data)) + '\n'
        txt = txt + 'max(data) = ' + str(np.max(self.data)) + '\n'
        return (txt)


class DistanceToVFeatureLayer(FeatureLayer):
    """
    Defines a distance to a state as a layer. This layer can then used for the calibration stage or the allocation stage::
    
        distance_to_2 = clumpy.definition.layer.LayerEV(name='elevation',time=0,scale=15)    
    
    However, it is recommended to prefer the case's method clumpy.definition.Case.add_distance_to_v_as_feature.
    
    Parameters
    ----------
    v : int
        The state to compute the distance from.
    
    layer_LUC : LandUseCoverLayer
        The land use cover used to compute the distance.
        
    name : string (default=``None``)
        The layer name. If none, name is defined as ``'distance_to_v_'+str(v)``.
    """
    
    def __init__(self, v, layer_LUC, name=None):
        
        if type(name)==type(None):
            name = 'distance_to_v_'+str(v)
        
        super().__init__(name=name, scale=layer_LUC.scale)
        self.v = v
        self.layer_LUC = layer_LUC
        self.update(layer_LUC=self.layer_LUC)
        layer_LUC.distance2v.append(self)

    def update(self, layer_LUC):
        v_matrix = (layer_LUC.data == self.v).astype(int)
        self.data = ndimage.distance_transform_edt(1 - v_matrix) * layer_LUC.scale


class TransitionProbabilityLayers():
    """
    Defines a set of :math:`P(vf|vi,z)` layers.
    
    Notes
    -----
        The set of layers is defined as a dictionary ``self.layers`` whose keys are :math:`(v_i,v_f)`.
    """
    
    def __init__(self):
        self.layers = {}
    
    def add_layer(self, vi, vf, data=None, path=None):
        """
        Add a layer to the set.        

        Parameters
        ----------
        vi : int
            initial state.
        vf : int
            final state.
        data : numpy array (default=None)
            The :math:`P(vf|vi,z)` data layer. If None, ``path`` is expected.
        path : string (default=None)
            The path to :math:`P(vf|vi,z)` tiff file. If None, ``data`` is expected.

        """
        self.layers[(vi, vf)] = _TransitionProbabilityLayer(vi, vf, data, path)
    
    def copy(self):
        """
        Make a copy.

        Returns
        -------
        c : TransitionProbabilityLayers
            A copy of the current object.

        """
        return(deepcopy(self))
        
    def export_all(self, path:str):
        """
        Export all layers as tif files through a zip archive file.

        Parameters
        ----------
        path : str
            Output zip file path. Required new folders are created without error raising.
            
        Notes
        -----
        All layer files inside the zip archive file are named as following : ``'P_vf' + str(vf) + '__vi' + str(vi) + '_z.tif'``.
        """
        files_names = []
        # folder_name = os.path.dirname(path)
        
        for layer in self.layers.values():
            files_names.append(layer.name+'.tif')
            layer.export_tiff(files_names[-1])
        
        command = 'zip .temp_P_vf__vi_z.zip'
        for file_name in files_names:
            command += ' '+file_name
        os.system(command)
        
        command = 'rm '
        for file_name in files_names:
            command += ' '+file_name
        os.system(command)
        
        # create folder if not exists
        folder_name = os.path.dirname(path)
        if not os.path.exists(folder_name) and folder_name!= '':
            os.makedirs(folder_name)
        
        os.system('mv .temp_P_vf__vi_z.zip '+path)
        
    def import_all(self, path:str):
        """
        Import all layers from a zip archive file.

        Parameters
        ----------
        path : str
            The zip file to import.
            
        Notes
        -----
        All layer files inside the zip archive file are expected to be named as following : ``'P_vf' + str(vf) + '__vi' + str(vi) + '_z.tif'``.

        """
        os.system('unzip '+path+' -d '+path+'.out')
        
        files = os.listdir(path+'.out/')
        
        self.layers = {}
        for file in files:
            start = 'vf'
            end = '__'
            vf = file[file.find(start)+len(start):file.rfind(end)]
            
            start = 'vi'
            end = '_z'
            vi = file[file.find(start)+len(start):file.rfind(end)]
            
            
            self.add_layer(vi=int(vi), vf=int(vf), path=path+'.out/'+file)
            
        os.system('rm -R '+path+'.out')
            

class _TransitionProbabilityLayer(_Layer):
    """
    Defines a P(vf|vi,z) layer. This layer can then used for the allocation stage.
    """

    def __init__(self, vi, vf, data=None, path=None):
        super().__init__('P_vf' + str(vf) + '__vi' + str(vi) + '_z')
        self.vi = vi
        self.vf = vf
        # self.Tif.layer_P_vf__vi_z = self

        if type(data) == np.ndarray:
            self.data = data

        if type(path) != type(None):
            self.import_tiff(path)

    def import_tiff(self, path):
        """
        Imports the layer data from a `tif` or a `tiff` file. The file extension can be something else than tiff. See `Image Pillow documentation, Image.open <https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.open>`_.
        
        :param path: path to the file
        :type path: str
        """
        print("importing tiff file '" + path + "'")
        self.path = path

        img = Image.open(path)  # image loading
        img.load()
        dtype = "float"
        self.data = np.asarray(img, dtype=dtype)  # conversion en numpy

        # self.get_dimensions()
        self.shape = np.shape(self.data)
        self.size = np.size(self.data)

        print("\t done, N_j=" + str(self.size))


