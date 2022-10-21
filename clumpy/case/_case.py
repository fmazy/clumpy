# -*- coding: utf-8 -*-

from .. import LandUseLayer, MaskLayer
from .. import start_log, stop_log
from .. import Palette, load_palette
from .. import load_transition_matrix

from .. import Territory

from datetime import datetime
import json
import logging


class Case():
    """
    Case object. It is the base object for configuring and running a LULC change modeling case with CLUMPY. It can also be used to set a simple case which can then be complexified through the API. No parameters are required for initializing the object. See 'load' function.
    """
    def __init__(self, regions=None):
        
        if regions is not None:
            self.regions = regions
        else:
            self.regions = []
        
    
    def __repr__(self):
        return 'Case('+str(self.get_regions_labels())+')'
    
    def get_regions_labels(self):
        """
        Get region labels as a list.
        
        Returns
        -------
        labels : list
        
        """
        
        return [r.label for r in self.regions]
    
    def get_regions_values(self):
        
        return [r.value for r in self.regions]
        
    def add_region(self, region):
        """
        Add a region. Nothing happen if the region is already in. 

        Parameters
        ----------
        region : Region

        Returns
        -------
        self

        """
        if region not in self.regions:
            if region.label in self.get_regions_labels():
                self.regions.append(region)
            else:
                Warning('The region is already in.')
        else:
            Warning('The region is already in.')
            
        return self
    
    def add_regions(self, regions):
        """
        Add a list of regions. Overwrites existing regions.

        Parameters
        ----------
        regions : list(Region)
            

        Returns
        -------
        Self

        """
        self.regions = regions
        return self
    
    def get_region_by_label(self, label):
        
        labels = self.get_regions_labels()
        return(self.regions[labels.index(label)])
        
    def get_region_by_value(self, value):
        
        values = self.get_regions_values()
        return(self.regions[values.index(value)])
    
    def get_region(self, info):
        if type(info) is str:
            return self.get_region_by_label(info)
        
        if type(info) is int:
            return self.get_region_by_value(info)
        
    
    # def open(self, info):
    #     """
    #     Loading function 

    #     Parameters
    #     ----------
    #     info : str or dict
    #         The load parameters. It can be a path (as a string) which points 
    #         to a json file, or a dict object directly.

    #     Returns
    #     -------
    #     self
        
    #     Attributes
    #     ----------
    #     params : dict
    #         The dict object containing all parameters.

    #     """
    #     if isinstance(info, dict):
    #         self._open_dict(info)
    #     elif isinstance(info, str):
    #         self._open_json(info)
    #     else:
    #         self.logger.error('Case/_case.py - Case.load() : Unexpected load info parameter.')
    
    # def _open_json(self, path):
    #     """
    #     load a json file
    #     """
    #     self.path = path
    #     # Opening JSON file
    #     f = open(path)
    
    #     # returns JSON object as
    #     # a dictionary
    #     params = json.load(f)
    
    #     # Closing file
    #     f.close()
        
    #     self._open_dict(params)
        
    #     return(self)
    
    # def _open_dict(self, params):
    #     """
    #     load a dict object
    #     """
        
    #     self.params = params
        
    #     output_folder = self.get_output_folder()
            
    #     self.logger = logging.getLogger('clumpy')
    #     self.logger.handlers = []
    #     self.logger.setLevel(logging.INFO)
    #     now = datetime.now()
    #     fh = logging.FileHandler(output_folder+now.strftime("log_%Y_%m_%d_%H_%M_%S.log"))
    #     fh.setLevel(logging.INFO)
    #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #     fh.setFormatter(formatter)
    #     self.logger.addHandler(fh)
        
    #     self.logger.info('Start Clumpy Case')
        
    #     return(self)
        
    
    # def save(self, path=None):
    #     """
    #     Save the parameters as a json file.

    #     Parameters
    #     ----------
    #     path : str, default=None
    #         The path to save the json file. If None, the current file is 
    #         overwritten !
    #     """
    #     if path is None:
    #         path = self.path
            
    #     f = open(path, "w")
    #     f.write(json.dumps(self.params, indent=4, sort_keys=False))
    #     f.close()
    
    # def save_as(self, path):
    #     """
    #     Save as a new json file.

    #     Parameters
    #     ----------
    #     path : str
    #          The path to save the json file.
    #     """
    #     self.save(path=path)
            
    # def fit(self):
    #     """
    #     Fit the case.
    #     """
    #     self.territory.fit(lul_initial = self.get_lul('initial'),
    #                        lul_final = self.get_lul('final'),
    #                        masks = self.get_regions_masks('calibration'))
        
    # def transition_probabilities(self):
    #     """
    #     Compute the transition probabilities of the case.
    #     """
        
    #     self.territory.transition_probabilities(\
    #             regions_transition_matrices = self.get_regions_transition_matrices(),
    #             lul = self.get_lul('start'),
    #             masks = self.get_regions_masks('allocation'),
    #             path_prefix = self.get_output_folder() + "proba")
        
    # def allocate(self):
    #     """
    #     Allocate the case.
    #     """
        
    #     self.territory.allocate(\
    #             regions_transition_matrices = self.get_regions_transition_matrices(),
    #             lul = self.get_lul('start'),
    #             masks = self.get_regions_masks('allocation'),
    #             path = self.get_output_folder() + "lul_output.tif",
    #             path_prefix_transition_probabilities = self.get_output_folder() + "proba")
    
    # def run(self):
    #     """
    #     Run the case according to the parameters.
    #     """
    #     self.logger.info('Start run function.')
    #     now = datetime.now()
    #     start_log(self.get_output_folder() + now.strftime("console_%Y_%m_%d_%H_%M_%S.md"))
        
    #     self.logger.info('calling fit function.')
    #     self.fit()
        
    #     if self.get_transition_probabilities_only():
    #         self.logger.info('calling transition_probabilities function.')
    #         self.transition_probabilities()
    #     else:
    #         self.logger.info('calling allocate function.')
    #         self.allocate()
        
    #     self.logger.info('End run function.')
    #     stop_log()
    
    # def make(self):
    #     """
    #     Make the case. This function must be called before any other operational function.
    #     """
    #     # the palette must be unchanged during all the process !
    #     self.palette = self.get_palette()
        
    #     self.territory = Territory(verbose=self.get_verbose())
    #     self.territory.make(self)
    
    # def get_palette(self):
    #     """
    #     Get the palette set in the parameters.
    #     """
    #     try:
    #         palette = load_palette(self.params.params['palette'])
    #         return(palette)
    #     except:
    #         self.logger.warning("Failed to open palette")
    #         return(Palette())
    #         raise
    
    # def get_transition_probabilities_only(self):
    #     """
    #     Get the transition probabilities only trigger parameter.
    #     """
    #     try:
    #         transition_probabilities_only = self.params.params['transition_probabilities_only']
    #         return(transition_probabilities_only)
    #     except:
    #         self.logger.warning("Failed to get transition_probability_only trigger. set True as default.")
    #         return(True)
    
    # def get_verbose(self):
    #     """
    #     Get the verbose parameter.
    #     """
    #     try:
    #         return(self.params.params['verbose'])
    #     except:
    #         self.logger.warning("Failed to get verbose trigger. set True as default.")
    #         return(True)
    
    # def get_output_folder(self):
    #     """
    #     Get the output folder parameter.
    #     """
    #     try:
    #         return(self.params.params['output_folder'])
    #     except:
    #         self.logger.warning("Failed to get output_folder string. set '' as default.")
    #         return('')
    
    # def get_lul(self, kind):
    #     """
    #     Get the land use layers parameter.
    #     """
    #     try:
    #         palette = self.palette
    #         return(LandUseLayer(path = self.params.params['lul_'+str(kind)],
    #                             palette=self.palette))
    #     except:
    #         self.logger.error("Failed to open land use layer at "+str(self.params['lul_'+str(kind)])+".")
    #         raise
    
    # def get_regions_masks(self, kind):
    #     """
    #     Get the masks parameters according to regions and the purpose 
    #     (calibration or allocation).
        
    #     Parameters
    #     ----------
    #     kind : {'calibration', 'allocation'}
    #         The purpose of the requested mask.
    #     """
    #     try:
    #         regions_masks = {}
    #         for region_label, region_params in self.params['regions'].items():
    #             if str(kind)+'_mask' in region_params.keys():
    #                 regions_masks[region_label] = MaskLayer(path=region_params[str(kind)+'_mask'])
    #         return(regions_masks)
    #     except:
    #         self.logger.warning("failed to load regions masks. returned None instead.")
    #         return(None)
    
    # def get_regions_transition_matrices(self):
    #     """
    #     Get the transition matrices parameters according to regions
    #     """
    #     try:
    #         regions_transition_matrices = {}
    #         for region_name, region_params in self.params['regions'].items():
    #             regions_transition_matrices[region_name] = \
    #                     load_transition_matrix(path=region_params['transition_matrix'],
    #                                            palette=self.palette)
    #         return(regions_transition_matrices)
    #     except:
    #         self.logger.error("Failed to open '"+str(region_params['transition_matrix'])+"'")
    #         raise
    
    # def new_region(self, label):
    #     if not isinstance(label, str):
    #         self.logger.error("The region label must be a string.")
    #         raise
        
    #     if label in self.params['regions'].keys():
    #         self.logger.error("The region label "+str(label)+" does already exist.")
    #         raise
    #     else:
    #         self.params['regions'][label] = {'transition_matrix':None,
    #                                          'calibration_mask':None,
    #                                          'allocation_mask':None}