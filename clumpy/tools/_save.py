#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 13:47:28 2021

@author: frem
"""

import os
import json
import numpy as np

_list_save_json_types = [str, int, list, float]

def _save_object(obj, path, excluded_keys=[]):
    attributes_file_name = 'attributes.json'
    
    # params
    files_names = []
    folder_name = os.path.dirname(path)
    
    json_params = {}
    
    for key, value in obj.__dict__.items():
        if key not in excluded_keys:
            if type(value) in _list_save_json_types:
                json_params[key] = value
            
            if type(value) is np.ndarray:
                np.save(key, value)
                files_names.append(key+'.npy')
    
    print(json_params)
    with open(attributes_file_name, 'w') as f:
        json.dump(json_params, f)
    
    files_names.append(attributes_file_name)
    
    # create output directory if needed
    if folder_name != "":
        os.system('mkdir -p ' + folder_name)
    
    # zip file
    command = 'zip '+path+' '
    for file_name in files_names:
        command += ' ' + file_name
    os.system(command)
    
    # remove files        
    command = 'rm '
    for file_name in files_names:
        command += ' ' + file_name
    os.system(command)
    
    return(True)

def _load_object(obj, path):
    os.system('unzip ' + path + ' -d ' + path + '.out')

    files = os.listdir(path + '.out/')

    for file in files:
        if file == 'attributes.json':
            f = open(path + '.out/attributes.json')
            params = json.load(f)

            for key, param in params.items():
                setattr(obj, key, param)

            f.close()

        elif file[-4:] == '.npy':
            setattr(obj, file[:-4], np.load(path + '.out/' + file))

    os.system('rm -R ' + path + '.out')
    return(True)

