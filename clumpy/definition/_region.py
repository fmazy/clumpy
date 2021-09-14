#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 17:45:24 2021

@author: frem
"""

class Region():
    def __init__(self,
                 name,
                 calibration_region,
                 allocation_region):
        
        self.name = name
        self.calibration_region = calibration_region
        self.allocation_region = allocation_region