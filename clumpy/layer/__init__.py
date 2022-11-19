# -*- coding: utf-8 -*-

from ._layer import Layer, convert_raster_file, get_J_W_Z, get_J_Z
from ._land_use_layer import LandUseLayer
from ._regions_layer import RegionsLayer
from ._ev_layer import EVLayer, get_bounds
from ._proba_layer import ProbaLayer

layers = {'layer': Layer,
          'land_use' : LandUseLayer,
          'ev' : EVLayer,
          'regions' : RegionsLayer,
          'proba' : ProbaLayer}

from ._io import open_layer