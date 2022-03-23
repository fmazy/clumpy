# from .. import layers as layers_dict
from .. import FeatureLayer
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
    

    # verbose
    # ========
    if 'verbose' in params.keys():
        verbose = params['verbose']

    # palette
    # ========
    if "palette" in params.keys():
        palette = load_palette(params['palette'])
    
    # regions
    # =======
    regions_features = {}
    regions_tm = {}
    
    if 'regions' in params.keys():
        for region_name, region_params in params['regions'].items():
            print('region', region_name)
            # features
            features = []
            if 'features' in region_params.keys():
                for feature_params in region_params['features']:
                    if feature_params['type'] == 'layer':
                        fp = extract_parameters(FeatureLayer, feature_params)
                        features.append(FeatureLayer(**fp))
                    elif feature_params['type'] == 'distance':
                        features.append(feature_params['state'])
            print(features)
            
            regions_features[region_name] = features
            
            # transition matrices
            if 'transition_matrix' in region_params.keys():
                regions_tm[region_name] = load_transition_matrix(region_params['transition_matrix'],
                                                                 palette=palette)
    
    territory = make_default_territory(transition_matrices=regions_tm,
                                        regions_features=regions_features,
                                        verbose=verbose)
            
    return (palette, regions_tm, territory)
    
    # load layers
    # ============
    # layers = {}
    # if "layers" in params.keys():
    #     for label, infos in params['layers'].items():

    #         layers_params = infos.copy()
    #         del layers_params['type']

    #         layers_params['path'] = data_folder_path + layers_params['path']

    #         if infos['type'] == 'land_use':
    #             layers_params['palette'] = palette


    #         layers[label] = layers_dict[infos['type']](**layers_params)

    # Transition matrices
    # # ====================
    # transition_matrices = {}
    # if "transition_matrices" in params.keys():
    #     for label, path in params['transition_matrices'].items():
    #         transition_matrices[label] = load_transition_matrix(path=data_folder_path + path,
    #                                                             palette=palette)
    
    # params['transition_matrices'] = transition_matrices
    
    # Features
    #=========
    # features = []
    # if 'features' in params.keys():
    #     for info in params['features']:
    #         if isinstance(info, str):
    #             features.append(layers[info])

    #         if isinstance(info, int):
    #             features.append(palette.get(info))
                
    #     params['features'] = features

    # Feature Selectors
    #==================
    
        
    # Territory
    # ==========
    territory = make_default_territory(**params)

    territory.check()

    return (palette, layers, transition_matrices, territory)
