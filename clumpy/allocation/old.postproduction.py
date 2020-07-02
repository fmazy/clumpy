"""
intro
"""

from scipy import ndimage # for the fill holes function
import numpy as np

from .. import definition

def fillHoles(T:definition.transition.Transition, map_f, overwrite=False):
    # processus pour corriger les trous
    # c'est bête et méchant
    # faudrait avoir de la mesure mais ça fait l'affaire pour l'instant
    print('\n fill holes process')
    
    map_f_out_data = map_f.data.copy()
    
    map_f_shape = np.shape(map_f.data)
    for Ti in T.Ti:
        for Tif in Ti.Tif:
            if Tif.alloc_fill_holes:
                print('\t '+str(Tif))
                map_vf = np.zeros(map_f_shape)
                map_vf.flat[np.where(map_f.data.flat==Tif.vf)[0]] = 1
                map_vf_filled = ndimage.morphology.binary_fill_holes(map_vf).astype(float)
                map_vf_filled_only_alloc = np.zeros(map_f_shape) # faut passer par ça pour proteger les pixels non vi qui pourrait être dans les trous
                map_vf_filled_only_alloc.flat[Ti.J_vi_pruned.j.values] = map_vf_filled.flat[Ti.J_vi_pruned.j.values]
                
                map_f_out_data = map_f_out_data * (map_vf_filled_only_alloc*(-1)+1) + map_vf_filled_only_alloc*Tif.vf
                
    if overwrite:
        map_f.data = map_f_out_data
        return map_f
    else:
        map_f_out = definition.layer.LayerLUC(name="result holes filled",
                               time=1,
                               scale=1)
        map_f_out.import_numpy(data=map_f_out_data)
        return(map_f_out)