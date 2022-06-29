# -*- coding: utf-8 -*-

import rasterio
from ..tools._path import path_split

from . import layers

def open_layer(path, kind='layer', keep_raster_obj=False, **kwargs):
    folder_path, file_name, file_ext = path_split(path)
    
    # read file
    try:
        raster = rasterio.open(path)
    except:
        raise(ValueError("Can't open land use layer file : "+str(path)))
    
    data = raster.read()
    
    geo_metadata = {'driver' : raster.driver,
                    'crs' : raster.crs,
                    'transform' : raster.transform}
    
    layer_class = layers[kind]
    
    if kind != 'proba' and len(data.shape)==3 and kind != 'layer':
        data = data.reshape(data.shape[1:])
    
    if kind == 'proba':
        final_states = [int(raster.tags(i_band)['final_state']) for i_band in range(1, data.shape[0]+1)]
        
        kwargs = dict(final_states=final_states)
    
    layer_obj = layer_class(data,
                       label=file_name,
                       geo_metadata = geo_metadata,
                       **kwargs)
    
    if keep_raster_obj:
        layer_obj._raster = raster
    
    return layer_obj
