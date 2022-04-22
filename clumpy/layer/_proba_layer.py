# -*- coding: utf-8 -*-

import numpy as np

from ._layer import Layer

class ProbaLayer(Layer):
    def __init__(self,
                 label=None,
                 time=0,
                 path=None,
                 data=None,
                 initial_states = None,
                 final_states = None,
                 copy_geo=None):
        
        band_tags = None
        
        if initial_states is not None and final_states is not None:
            band_tags = [{'initial_state' : initial_states[i],
                          'final_state' : final_states[i]} for i in range(len(initial_states))]
        
        super().__init__(label=label,
                         time=time,
                         path=path,
                         data=data,
                         band_tags=band_tags,
                         copy_geo=copy_geo)
    
    def get_band_idx_of_initial_state(self, initial_state):
        tags = self.get_band_tags()
        band_idx = []
        final_states = []
        
        for i in range(len(tags)):
            if int(tags[i]['initial_state']) == int(initial_state):
                band_idx.append(i + 1)
                final_states.append(int(tags[i]['final_state']))
        
        return(band_idx, final_states)
    
    def yield_proba_of_initial_state(self, initial_state):
        band_idx, final_states = self.get_band_idx_of_initial_state(initial_state=initial_state)
        
        for i in range(len(band_idx)):
            yield final_states[i], self.get_data(band=band_idx[i])
    
    def get_proba(self, 
                  initial_state, 
                  final_state):
        n_bands = self.get_n_bands()
        
        for i_band in range(1, n_bands+1):
            if int(self.raster_.tags(i_band)['initial_state']) == int(initial_state) and\
            int(self.raster_.tags(i_band)['final_state']) == int(final_state):
                return(self.get_data(i_band))
        
        return(None)
            
def create_proba_layer(path,
                       lul,
                       p):        
    shape = lul.get_data().shape
    
    M = np.array([]).reshape((0,) + lul.get_data().shape)
    initial_states = []
    final_states = []
    
    initial_final = []
        
    for region_label, p_region in p.items():
        for initial_state, p_land in p_region.items():
            J__land, P_v__u_Y__land, final_states__land = p_land
            
            M__land = _get_proba_layer_data(J = J__land,
                                            P_v__u_Y = P_v__u_Y__land,
                                            shape = shape)
                        
            for i in range(len(final_states__land)):
                initial_final__i = (initial_state,
                                    final_states__land[i])
                if initial_final__i in initial_final:
                    i_band = initial_final.index(initial_final__i)
                    M[i_band] += M__land[i]
                else:
                    M = np.concatenate((M, M__land[[i]]))
                    initial_final.append(initial_final__i)
    
    initial_states = [initial for initial, final in initial_final]
    final_states = [final for initial, final in initial_final]
    
    proba_layer = ProbaLayer(path=path,
                            data=M,
                            initial_states = initial_states,
                            final_states = final_states,
                            copy_geo=lul)
        
    return(proba_layer)

def _get_proba_layer_data(J,
                          P_v__u_Y,
                          shape):
    n_bands = P_v__u_Y.shape[1]
    M = np.zeros((n_bands,) + shape)
    
    for i_band in range(n_bands):
        M[i_band].flat[J] = P_v__u_Y[:, i_band]
        
    return(M)