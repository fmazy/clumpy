# -*- coding: utf-8 -*-

from ._layer import Layer, convert_raster_file
from ._land_use_layer import LandUseLayer
from ._mask_layer import MaskLayer
from ._feature_layer import FeatureLayer
from ._proba_layer import ProbaLayer, create_proba_layer

layers = {'land_use' : LandUseLayer,
          'feature' : FeatureLayer,
          'mask' : MaskLayer,
          'proba' : ProbaLayer}