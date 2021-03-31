# -*- coding: utf-8 -*-

# import numpy as np
import numpy as np
from scipy import ndimage
import time

from ._layer import FeatureLayer
# from ..utils import ndarray_suitable_integer_type

class Case():
    """A land use and cover change model case.
    
    Parameters
    ----------
    params : dict
        parameters, see example and user guide.
        
    """
    def __init__(self, params):
        self.params = params
    
    def make(self, initial_luc_layer, final_luc_layer=None, verbose=0):
        """Make the case

        Parameters
        ----------
        initial_luc_layer : :class:`clumpy.definition.LandUseCoverLayer`
            The initial LUC layer.
        final_luc_layer : :class:`clumpy.definition.LandUseCoverLayer`, default=None
            The final LUC layer. This parameter is optional.

        Returns
        -------
        X_u : dict of :class:`numpy.ndarray` of shape (n_samples, n_features)
            A dict with :math:`u` keys. Each dict values are features data, where n_samples is the number of samples and n_features is the number of features.
        
        (v_u) : dict of :class:`numpy.ndarray` of shape (n_samples,)
            Returned if `final_luc_layer` is provided. A dict with :math:`u` keys. Each dict values are target values, *i.e.* :math:`v`, where n_samples is the number of samples.
            
        Examples
        --------
        ::
            
            params = {3:{'v':[2,7],
                         'features':[('layer',dem),
                                     ('distance',2),
                                     ('distance',7)]
                         },
                      4:{'v':[2,3],
                         'features':[('layer',dem),
                                     ('distance',2),
                                     ('distance',3)]}
                      }
            case = clumpy.definition.Case(params)
            X_u, v_u = case.make(LUC1998, LUC2003)

        """
        
        start_time=time.time()
        
        # check the case
        if verbose > 0:
            print('case checking...')
        check_case(self)
        
        
        # first compute distances
        distances = {}
        
        if verbose > 0:
            print('distances computing')
            print('===================')
        # for each u
        for u in self.params.keys():
            # for each feature
            for feature_type, info in self.params[u]['features']:
                # if it is a distance
                if feature_type == 'distance':
                    # if this distance has not been computed yet
                    if info not in distances.keys():
                        if verbose > 0:
                            print('\t distance to '+str(info)+'...')
                        # make bool v matrix
                        v_matrix = (initial_luc_layer.raster_.read(1) == info).astype(int)
                        # compute distance
                        # should get scale value from the tiff file.
                        distances[info] = ndimage.distance_transform_edt(1 - v_matrix)
        
        if verbose > 0:
            print('sets creating')
            print('=============')
        
        # initialize X_u                    
        X_u = {}
        
        # if final luc layer, initialize v_u
        if final_luc_layer is not None:
            v_u = {}
        
        # for each u
        for u in self.params.keys():
            if verbose > 0:
                print('\t u='+str(u)+'...')
            
            # get pixels indexes whose initial states are u
            # J = ndarray_suitable_integer_type(np.where(initial_luc_layer.raster_.read(1).flat==u)[0])
            J = np.where(initial_luc_layer.raster_.read(1).flat==u)[0]
            
            
            # create feature names
            for feature_type, info in self.params[u]['features']:
                # switch according z_type

                if feature_type == 'layer' or feature_type == 'binary_layer':
                    # just get data
                    x = info.raster_.read(1).flat[J]
                
                elif feature_type == 'distance':
                    # get distance data
                    x = distances[info].flat[J]
                
                elif feature_type == 'numpy':
                    # just get data
                    x = info
                    
                # if X_u is not yet defined
                if u not in X_u.keys():
                    X_u[u] = x
                
                # else column stack
                else:
                    X_u[u] = np.column_stack((X_u[u], x))

            # if only one feature, reshape X as a column
            if len(self.params[u]['features']) == 1:
                X_u[u] = X_u[u][:,None]
            
            # if final luc layer
            if final_luc_layer is not None:
                # just get data
                v_u[u] = final_luc_layer.raster_.read(1).flat[J]
                
                if 'v' in self.params[u].keys():
                    v_u[u][~np.isin(v_u[u], self.params[u]['v'])] = u
        
        self.creating_time_ = time.time()-start_time
        
        if verbose > 0:
            print('case creating is a success !')
            print('creating time: '+str(round(self.creating_time_,2))+'s')

        # if no final luc layer
        if final_luc_layer is None:
            return(X_u)
        
        else:
            return(X_u, v_u)

def check_case(case):
    """
    Raise an error if the case params are uncorrect.

    Parameters
    ----------
    case : :class:`clumpy.definition.Case`
        A clumpy case.
    """
    
    if type(case.params) is not dict:
        raise(TypeError("Case params should be a dict. See documentation for examples."))
    
    for u in case.params.keys():
        if type(u) is not int:
            raise(TypeError("Case params keys should be integers. See documentation for examples."))
            
        if type(case.params[u]) is not dict:
            raise(TypeError("case.params["+str(u)+"] should be a dict. See documentation for examples."))
            
        for key, value in case.params[u].items():
            
            if key == 'v':
                for v in value:
                    if type(v) is not int:
                        raise(TypeError("case.params["+str(u)+"]['v'] should be a list of integers. See documentation for examples."))
                    
            elif key == 'features':
                if type(value) is not list:
                    raise(TypeError("case.params["+str(u)+"]['features'] should be a list of tuples. See documentation for examples."))
                
                for idx, feature in enumerate(value):
                    if type(feature) is not tuple:
                        raise(TypeError("case.params["+str(u)+"]['features'] should be a list of tuples. See documentation for examples."))
                    
                    if feature[0] == 'layer' or feature[0] == 'binary_layer':
                        if type(feature[1]) is not FeatureLayer:
                            raise(TypeError("case.params["+str(u)+"]['features']["+str(idx)+"][1] should be a clumpy.definition.FeatureLayer object. See documentation for examples."))
                            
                    elif feature[0] == 'distance':
                        if type(feature[1]) is not int:
                            raise(TypeError("case.params["+str(u)+"]['features']["+str(idx)+"][1] should be an integer. See documentation for examples."))
                            
                    elif feature[0] == 'numpy':
                        if type(feature[1]) is not np.ndarray:
                            raise(TypeError("case.params["+str(u)+"]['features']["+str(idx)+"][1] should be a ndarray. See documentation for examples."))

                    else:
                        raise(ValueError("case.params["+str(u)+"]['features']["+str(idx)+"] is expected to be {'layer', 'distance', 'numpy'. See documentation for examples.}"))
            
            else:
                raise(ValueError("case.params["+str(u)+"] keys are expected to be {'v', 'features'}. See documentation for examples."))
    

