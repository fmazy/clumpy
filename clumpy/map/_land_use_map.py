# -*- coding: utf-8 -*-
import numpy as np
from scipy import ndimage

from ._map import Map

class LandUseMap(Map):
    
    def __init__(self):
        self.distances = {}
    
    