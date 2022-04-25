# -*- coding: utf-8 -*-

import rasterio
from ..tools._path import path_split

from . import layers

def open_layer(path, kind='layer', **kwargs):
    folder_path, file_name, file_ext = path_split(path)
    
    # read file
    try:
        raster = rasterio.open(path)
    except:
        raise(ValueError("Can't open land use layer file : "+str(path)))
    
    data = raster.read()
    
    band_tags = [raster.tags(i_band) for i_band in range(1, data.shape[0]+1)]
    
    geo_metadata = {'driver' : raster.driver,
                    'crs' : raster.crs,
                    'transform' : raster.transform}
    
    layer_class = layers[kind]
    
    return layer_class(data,
                       label=file_name,
                       band_tags = band_tags,
                       geo_metadata = geo_metadata,
                       **kwargs)