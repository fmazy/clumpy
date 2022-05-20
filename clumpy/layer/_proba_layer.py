# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy
from ._layer import Layer

class ProbaLayer(Layer):

    def __new__(cls, 
                input_array,
                final_states,
                label=None,
                dtype=None,
                geo_metadata=None):
        
        obj = super().__new__(cls, 
                              input_array,
                              label=label,
                              dtype=dtype,
                              geo_metadata=geo_metadata)
        
        obj.final_states = final_states
        
        return obj      
    
    def copy(self):
        return ProbaLayer(np.array(self),
                          label=self.label,
                          geo_metadata=deepcopy(self.geo_metadata))
    
    def get_proba(self, final_state):
        return self[self.final_states.index(final_state)]
    
    def yield_proba(self):
        for i, final_state in enumerate(self.final_states):
            yield final_state, self.get_proba(final_state)
    
    def fusion(self,
               proba_layer):
        
        M = np.array(self)
        
        final_states = deepcopy(self.final_states)
        
        for i_prime, final_state_prime in enumerate(proba_layer.final_states):
            if final_state_prime in final_states:
                i = final_states.index(final_state_prime)
                M[i] += proba_layer[i_prime]
            else:
                final_states.append(final_state_prime)
                M = np.concatenate((M, proba_layer[[i_prime]]))
                
        return ProbaLayer(M,
                          final_states=final_states,
                          label=self.label + '_fusion',
                          geo_metadata = deepcopy(self.geo_metadata))
    
    def save(self, path):
        band_tags = [{'final_state': final_state} for final_state in self.final_states]
        
        self._save(path=path,
                   band_tags=band_tags)
    
    def get_flat_proba(self, J):
        return np.vstack([P.flat[J] for P in self]).T

def create_proba_layer(J,
                       P,
                       final_states,
                       shape,
                       geo_metadata=None):
    M = np.zeros((len(final_states),) + shape)
    
    for i, final_state in enumerate(final_states):
        M[i].flat[J] = P[:, i]
    
    proba_layer = ProbaLayer(M,
                             final_states=final_states,
                             label='proba',
                             geo_metadata = deepcopy(geo_metadata))
    
    return proba_layer