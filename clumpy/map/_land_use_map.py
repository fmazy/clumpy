# -*- coding: utf-8 -*-
import numpy as np
from scipy import ndimage

from ._map import Map

class LandUseMap(Map):
    
    def __init__(self):
        self.distances = {}
    
    def get_J(self,
              state,
              mask=None):
        """
        """    
        # get pixels indexes whose initial states are u
        # within the mask
        return np.all((np.where(self.flat == int(state))[0],
                       mask.flat))
    
    def get_V(self,
              J,
              final_states=None):
                
        V = self.flat[J]
        
        if final_states is None:
            return(V)
        
        else:
            idx = np.isin(V, self.final_states)
            J = J[idx]
            V = V[idx]
            
            return(V, J)
    
    def get_X(self, 
              J,
              features,
              distances_to_states={}):
        
        X = None
        
        for info in features:
            # switch according z_type
            if isinstance(info, Layer):
                # just get data
                x = info.get_data().flat[J]

            elif isinstance(info, int):
                # get distance data
                if info not in distances_to_states.keys():
                    _compute_distance(info, lul.get_data(), distances_to_states)
                    
                x = distances_to_states[info].flat[J]
                
            else:
                logger.error('Unexpected feature info : ' + type(info) + '. Occured in \'_base/_land.py, Land.get_values()\'.')
                raise (TypeError('Unexpected feature info : ' + type(info) + '.'))

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
    
    def compute_distance(state):
        v_matrix = (self == int(state)).astype(int)
        self.distances[int(state)] = ndimage.distance_transform_edt(1 - v_matrix)