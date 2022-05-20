# -*- coding: utf-8 -*-

from ._layer import Layer, convert_raster_file
from ._land_use_layer import LandUseLayer
from ._mask_layer import MaskLayer
from ._feature_layer import FeatureLayer
from ._proba_layer import ProbaLayer

layers = {'layer': Layer,
          'land_use' : LandUseLayer,
          'feature' : FeatureLayer,
          'mask' : MaskLayer,
          'proba' : ProbaLayer}

from ._io import open_layer