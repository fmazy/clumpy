# -*- coding: utf-8 -*-

from .. import FeatureLayer, LandUseLayer, MaskLayer
from .. import start_log, stop_log
from .. import Palette, load_palette
from .. import load_transition_matrix
from ._make import make_default_territory
from ..tools._funcs import extract_parameters

from .. import Territory, Region

from datetime import datetime
import json
import rasterio
import logging


class Case():
    def __init__(self):
        self.territory = None
        self.params = None
    
    def load(self, path):
        self.path = path
        # Opening JSON file
        f = open(path)
    
        # returns JSON object as
        # a dictionary
        self.params = json.load(f)
    
        # Closing file
        f.close()
    
        # OUTPUT
        # ======
        
        output_folder = self.get_output_folder()
            
        self.logger = logging.getLogger('clumpy')
        self.logger.handlers = []
        self.logger.setLevel(logging.INFO)
        now = datetime.now()
        fh = logging.FileHandler(output_folder+now.strftime("log_%Y_%m_%d_%H_%M_%S.log"))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        self.logger.info('Start Clumpy Case')
        
        return(self)
    
    def save(self, path=None):
        if path is None:
            path = self.path
            
        f = open(path, "w")
        f.write(json.dumps(self.params, indent=4, sort_keys=False))
        f.close()
    
    def save_as(self, path):
        self.save(path=path)
            
    def fit(self):
        self.territory.fit(lul_initial = self.get_lul('initial'),
                           lul_final = self.get_lul('final'),
                           masks = self.get_regions_masks('calibration'))
        
    def transition_probabilities(self):
        self.territory.transition_probabilities(\
                regions_transition_matrices = self.get_regions_transition_matrices(),
                lul = self.get_lul('start'),
                masks = self.get_regions_masks('allocation'),
                path_prefix = self.get_output_folder() + "proba")
        
    def allocate(self):
        self.territory.allocate(\
                regions_transition_matrices = self.get_regions_transition_matrices(),
                lul = self.get_lul('start'),
                masks = self.get_regions_masks('allocation'),
                path = self.get_output_folder() + "lul_output.tif",
                path_prefix_transition_probabilities = self.get_output_folder() + "proba")
    
    def run(self):
        self.logger.info('Start run function.')
        now = datetime.now()
        start_log(self.get_output_folder() + now.strftime("console_%Y_%m_%d_%H_%M_%S.md"))
        
        self.logger.info('calling fit function.')
        self.fit()
        
        if self.get_transition_probabilities_only():
            self.logger.info('calling transition_probabilities function.')
            self.transition_probabilities()
        else:
            self.logger.info('calling allocate function.')
            self.allocate()
        
        self.logger.info('End run function.')
        stop_log()
    
    def make(self):
        # the palette must be unchanged during all the process !
        self.palette = self.get_palette()
        
        self.territory = Territory(verbose=self.get_verbose())
        self.territory.make(self)
    
    def get_palette(self):
        try:
            palette = load_palette(self.params['palette'])
            return(palette)
        except:
            self.logger.warning("Failed to open palette")
            return(Palette())
            raise
    
    def get_transition_probabilities_only(self):
        try:
            transition_probabilities_only = self.params['transition_probabilities_only']
            return(transition_probabilities_only)
        except:
            self.logger.warning("Failed to get transition_probability_only trigger. set True as default.")
            return(True)
    
    def get_verbose(self):
        try:
            return(self.params['verbose'])
        except:
            self.logger.warning("Failed to get verbose trigger. set True as default.")
            return(True)
    
    def get_output_folder(self):
        try:
            return(self.params['output_folder'])
        except:
            self.logger.warning("Failed to get output_folder string. set '' as default.")
            return('')
    
    def get_lul(self, kind):
        try:
            palette = self.palette
            return(LandUseLayer(path = self.params['lul_'+str(kind)],
                                palette=self.palette))
        except:
            self.logger.error("Failed to open land use layer at "+str(self.params['lul_'+str(kind)])+".")
            raise
    
    def get_regions_masks(self, kind):
        try:
            regions_masks = {}
            for region_label, region_params in self.params['regions'].items():
                if str(kind)+'_mask' in region_params.keys():
                    regions_masks[region_label] = MaskLayer(path=region_params[str(kind)+'_mask'])
            return(regions_masks)
        except:
            self.logger.warning("failed to load regions masks. returned None instead.")
            return(None)
    
    def get_regions_transition_matrices(self):
        try:
            regions_transition_matrices = {}
            for region_name, region_params in self.params['regions'].items():
                regions_transition_matrices[region_name] = \
                        load_transition_matrix(path=region_params['transition_matrix'],
                                               palette=self.palette)
            return(regions_transition_matrices)
        except:
            self.logger.error("Failed to open '"+str(region_params['transition_matrix'])+"'")
            raise
    