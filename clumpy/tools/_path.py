#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 09:53:29 2021

@author: frem
"""

import os

def path_split(path, prefix=False):
    """
    Split a path.
    
    Parameters
    ----------
    
    path : str
        The path.
    
    Returns
    -------
    folder_path : str
        The folder path.
    file_name : str
        The file name without extension.
    file_ext : str
        The file extension.
    """
    folder_path = os.path.dirname(path)
    
    base_name = os.path.basename(path)
    
    if prefix:
        return(folder_path, base_name)
    
    
    base_name_splitted = base_name.split('.')
    
    file_ext = base_name_splitted[-1]
    file_name = ''
    for b in base_name_splitted[:-1]:
        file_name += b
    
    return(folder_path, file_name, file_ext)

def create_directories(folder_path):
    if len(folder_path) > 0:
        if os.path.exists(folder_path) == False:
            os.system('mkdir -p ' + folder_path)