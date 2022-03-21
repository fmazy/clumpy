from .. import layers as layers_dict
from .. import start_log
from .. import load_palette
from .. import load_transition_matrix
from ._make import make_default_territory


import json


def load(path):
    # Opening JSON file
    f = open(path)

    # returns JSON object as
    # a dictionary
    json_case = json.load(f)

    # Closing file
    f.close()

    # data folder path
    # =================
    if 'data_folder_path' in json_case.keys():
        data_folder_path = json_case['data_folder_path']
    else:
        data_folder_path = ''

    # log file
    # =========
    if 'log_file' in json_case.keys():
        start_log(data_folder_path + json_case['log_file'])

    # verbose
    # ========
    if 'verbose' in json_case.keys():
        verbose = json_case['verbose']

    # palette
    # ========
    if "palette" in json_case.keys():
        palette = load_palette(data_folder_path + json_case['palette'])

    # load layers
    # ============
    layers = {}
    if "layers" in json_case.keys():
        for label, infos in json_case['layers'].items():

            layers_params = infos.copy()
            del layers_params['type']

            layers_params['path'] = data_folder_path + layers_params['path']

            if infos['type'] == 'land_use':
                layers_params['palette'] = palette


            layers[label] = layers_dict[infos['type']](**layers_params)

    # Transition matrices
    # ====================
    transition_matrices = {}
    if "transition_matrices" in json_case.keys():
        for label, path in json_case['transition_matrices'].items():
            transition_matrices[label] = load_transition_matrix(path=data_folder_path + path,
                                                                palette=palette)

    # Features
    #=========
    features = []
    if 'features' in json_case.keys():
        for info in json_case['features']:
            if isinstance(info, str):
                features.append(layers[info])

            if isinstance(info, int):
                features.append(palette.get(info))

    # Feature Selectors
    #==================
    if 'feature_selector' in json_case.keys():
        feature_selector = json_case['feature_selector']
    else:
        feature_selector = []

    # Territory
    # ==========
    make_parameters = {'n_jobs_predict':1,
                       'n_jobs_neighbors':1,
                       'n_fit_max':2*10**4,
                       'n_predict_max':2*10**4,
                       'density_estimation_method':'kde',
                       'q':51,
                       'kernel':'box',
                       'P_v_min':0.0,
                       'n_samples_min':1,
                       'update_P_Y':False,
                       'n_allocation_try':1000,
                       'fit_bootstrap_patches':True}

    for key, default in make_parameters.items():
        if key in json_case.keys():
            make_parameters[key] = json_case[key]
    
    print(make_parameters)
    territory = make_default_territory(transition_matrices=transition_matrices,
                                       features = features,
                                       feature_selector = feature_selector,
                                       **make_parameters)

    territory.check()

    return (palette, layers, transition_matrices, territory)
