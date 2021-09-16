#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 07:40:38 2021

@author: frem
"""

class Model():
    """
    Model parameters
    
    """
    def __init__(self, regions = []):
        self.regions = regions
        
    def __repr(self):
        return('model_params')