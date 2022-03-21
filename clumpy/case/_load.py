from .. import layers as layers_dict
from .. import start_log
from .. import load_palette
from .. import load_transition_matrix
from ._make import make_default_territory
from ..tools._funcs import extract_parameters

import json


def load(path):
    # Opening JSON file
    f = open(path)

    # returns JSON object as
    # a dictionary
    params = json.load(f)

    # Closing file
    f.close()
    
    
    # data folder path
    # =================
    if 'data_folder_path' in params.keys():
        data_folder_path = params['data_folder_path']
    else:
        data_folder_path = ''

    # log file
    # =========
    if 'log_file' in params.keys():
        start_log(data_folder_path + params['log_file'])

    # verbose
    # ========
    if 'verbose' in params.keys():
        verbose = params['verbose']

    # palette
    # ========
    if "palette" in params.keys():
        palette = load_palette(data_folder_path + params['palette'])

    # load layers
    # ============
    layers = {}
    if "layers" in params.keys():
        for label, infos in params['layers'].items():

            layers_params = infos.copy()
            del layers_params['type']

            layers_params['path'] = data_folder_path + layers_params['path']

            if infos['type'] == 'land_use':
                layers_params['palette'] = palette


            layers[label] = layers_dict[infos['type']](**layers_params)

    # Transition matrices
    # ====================
    transition_matrices = {}
    if "transition_matrices" in params.keys():
        for label, path in params['transition_matrices'].items():
            transition_matrices[label] = load_transition_matrix(path=data_folder_path + path,
                                                                palette=palette)
    
    params['transition_matrices'] = transition_matrices
    
    # Features
    #=========
    features = []
    if 'features' in params.keys():
        for info in params['features']:
            if isinstance(info, str):
                features.append(layers[info])

            if isinstance(info, int):
                features.append(palette.get(info))
                
        params['features'] = features

    # Feature Selectors
    #==================
    
        
    # Territory
    # ==========
    territory = make_default_territory(**params)

    territory.check()

    return (palette, layers, transition_matrices, territory)
