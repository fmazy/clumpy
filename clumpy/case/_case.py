# -*- coding: utf-8 -*-

from .. import FeatureLayer, LandUseLayer, MaskLayer
from .. import start_log, stop_log
from .. import load_palette
from .. import load_transition_matrix
from ._make import make_default_territory
from ..tools._funcs import extract_parameters

from datetime import datetime
import json
import rasterio


class Case():
    def __init__(self):
        self.territory = None
        self.palette = None
        self.lul_initial = None
        self.lul_final = None
        self.lul_start = None
        self.regions_features = None
        self.regions_tm = None
        self.regions_masks_calib = None
        self.regions_masks_alloc = None
        self.verbose = 0
        self.output_folder = None
        self.transition_probabilities_only = False
    
    def load(self, path):
        # Opening JSON file
        f = open(path)
    
        # returns JSON object as
        # a dictionary
        self.params = json.load(f)
    
        # Closing file
        f.close()
        
    
        # OUTPUT
        # ======
        if 'verbose' in self.params.keys():
            self.verbose = self.params['verbose']
        if 'output_folder' in self.params.keys():
            self.output_folder = self.params['output_folder']
        if 'transition_probabilities_only' in self.params.keys():
            self.transition_probabilities_only = self.params['transition_probabilities_only']
        
        # palette
        # ========
        if "palette" in self.params.keys():
            self.palette = load_palette(self.params['palette'])
        
        # regions
        # =======
        self.regions_features = {}
        self.regions_transition_matrices = {}
        self.regions_masks_calib = {}
        self.regions_masks_alloc = {}
        
        if 'regions' in self.params.keys():
            for region_name, region_params in self.params['regions'].items():
                print('region', region_name)
                # features
                features = []
                if 'features' in region_params.keys():
                    for feature_params in region_params['features']:
                        if feature_params['type'] == 'layer':
                            fp = extract_parameters(FeatureLayer, feature_params)
                            features.append(FeatureLayer(**fp))
                        elif feature_params['type'] == 'distance':
                            features.append(self.palette._get_by_value(feature_params['state']))
                print(features)
                
                self.regions_features[region_name] = features
                
                # transition matrices
                if 'transition_matrix' in region_params.keys():
                    self.regions_transition_matrices[region_name] = \
                                load_transition_matrix(path=region_params['transition_matrix'],
                                                       palette=self.palette)
                                
                # masks calib
                if 'mask_calib' in region_params.keys():
                    self.regions_masks_calib[region_name] = \
                                MaskLayer(path=region_params['mask_calib'])
                
                # masks calib
                if 'mask_alloc' in region_params.keys():
                    self.regions_masks_alloc[region_name] = \
                                MaskLayer(path=region_params['mask_alloc'])
        
        self.territory = make_default_territory(transition_matrices=self.regions_transition_matrices,
                                                regions_features=self.regions_features,
                                                verbose=self.verbose)
        
        # land use layers
        # ===============
        self.lul_initial = LandUseLayer(path = self.params['lul_initial'],
                                        palette=self.palette)
        self.lul_final = LandUseLayer(path = self.params['lul_final'],
                                      palette=self.palette)
        self.lul_start = LandUseLayer(path = self.params['lul_start'],
                                      palette=self.palette)
        
        
        
        return(self)
    
    def fit(self):
        self.territory.fit(lul_initial = self.lul_initial,
                           lul_final = self.lul_final,
                           masks = self.regions_masks_calib)
        
    def transition_probabilities(self):
        self.territory.transition_probabilities(\
                regions_transition_matrices = self.regions_transition_matrices,
                lul = self.lul_start,
                masks = self.regions_masks_alloc,
                path_prefix = self.output_folder + "proba")
        
    def allocate(self):
        self.territory.allocate(\
                regions_transition_matrices = self.regions_transition_matrices,
                lul = self.lul_start,
                masks = self.regions_masks_alloc,
                path = self.output_folder + "lul_output.tif",
                path_prefix_transition_probabilities = self.output_folder + "proba")
    
    def run(self):
        now = datetime.now()
        start_log(self.output_folder + now.strftime("log_%Y_%m_%d_%H_%M_%S.md"))
                
        self.fit()
        
        if self.transition_probabilities_only:
                self.transition_probabilities()
        else:
            self.allocate()
        stop_log()