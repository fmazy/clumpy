# -*- coding: utf-8 -*-
from copy import deepcopy
from scipy import ndimage
import numpy as np

from ._layer import Layer, FeatureLayer
from ._state import State
from ..feature_selection import Pipeline

class Features():
    def __init__(self, 
                 f_list = [],
                 selector = None):
        self.list = []
        for item in f_list:
            if isinstance(item, Layer):
                self.list.append(item)
            else:
                self.list.append(int(item))
        
        self.selector = selector
        self._fitted = False
    
    def __repr__(self):
        return str(self.list)+', '+str(self.selector)
    
    def __iter__(self):
        for feature in self.list:
            yield feature

    def __len__(self):
        return (len(self.list))
    
    def __getitem__(self, i):
         return self.list[i]
    
    def copy(self):
        return(Features(f_list = [f for f in self.list],
                        selector = deepcopy(self.selector)))
    
    def check(self, objects=[]):
        if self.selector in objects:
            raise(ValueError("Selector objects must be different."))
        else:
            objects.append(self.selector)
        
        if isinstance(self.selector, Pipeline):
            self.selector.check(objects=objects)
    
    def get_all(self, 
                J,
                lul,
                distances_to_states={}):
        
        X = None
                
        for info in self.list:
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
    
    def fit_selector(self, X, V):
        self.selector.fit(X=X, y=V)
        
        self._fitted = True
        
        return(self)
    
    def select(self, X):
        if not self._fitted:
            raise(TypeError("The feature object has to be fitted before calling select()."))
        return(self.selector.transform(X))

    def get(self,
            J,
            lul,
            distances_to_states={}):
        
        X = self.get_all(J=J,
                         lul=lul,
                         distances_to_states=distances_to_states)
        return(self.select(X))

    def fit(self,
            J, 
            V,
            state,
            lul, 
            distances_to_states={},
            return_X=False):
                
        X = self.get_all(J=J,
                                  lul=lul,
                                  distances_to_states=distances_to_states)
        self.fit_selector(X, V)
        
        if return_X:
            return(self.select(X))
        else:
            return(self)               
    
    def get_selected_features(self):
        if not self._fitted:
            raise(TypeError("The feature object has to be fitted before calling get_selected_features()."))
        
        return([self.list[i] for i in self.selector._cols_support])
    
    def get_bounds(self):
        selected_features = self.get_selected_features()
        
        bounds = []
        for id_col, item in enumerate(selected_features):
            if isinstance(item, FeatureLayer):
                if item.bounded in ['left', 'right', 'both']:
                    # one takes as parameter the column id of
                    # bounded features AFTER feature selection !
                    bounds.append((id_col, item.bounded))
                    
            # if it is a state distance, add a low bound set to 0.0
            if isinstance(item, State) or isinstance(item, int):
                bounds.append((id_col, 'left'))
        
        return(bounds)

        

def _compute_distance(state_value, data, distances_to_states):
    v_matrix = (data == state_value).astype(int)
    distances_to_states[state_value] = ndimage.distance_transform_edt(1 - v_matrix)