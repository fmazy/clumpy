# -*- coding: utf-8 -*-

from .. import LandUseLayer, RegionsLayer, EVLayer
from .. import start_log, stop_log
from .. import Palette, load_palette
from .. import load_transition_matrix

from ..layer._proba_layer import create_proba_layer

from ..ev_selection import EVSelectors

from .. import open_layer
from ..layer import get_bounds, get_J_W_Z, get_J_Z

# from .. import Territory

from datetime import datetime
import json
import logging

import numpy as np

class Case():
    """
    Case object. It is the base object for configuring and running a LULC change modeling case with CLUMPY. It can also be used to set a simple case which can then be complexified through the API. No parameters are required for initializing the object. See 'load' function.
    """
    def __init__(self,
                 regions=None,
                 palette=None,
                 verbose=1):
        
        if regions is not None:
            self.regions = regions
        else:
            self.regions = []
        
        self.palette = palette
        
        self.verbose = verbose
    
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
            if region.value not in self.get_regions_values():
                if region.label not in self.get_regions_labels():
                    self.regions.append(region)
                else:
                    Warning('The region label is already in.')
            else:
                Warning('The region value is already in.')
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
    
    def set_palette(self, palette):
        self.palette = palette
    
    def get_ev_labels(self, evs):
        ev_labels = []
        for ev in evs:
            if isinstance(ev, EVLayer):
                ev_labels.append(ev.label)
            elif type(ev) is int:
                state_label = self.palette.get(ev).label
                ev_labels.append('dist. to '+state_label)
        return ev_labels
    
    def init_ev_selectors(self, evs):
        ev_labels = self.get_ev_labels(evs=evs)
        for region in self.regions:
            for land in region.lands:
                for i, v in enumerate(land.final_states):
                    if land.ev_selectors is not None:
                        ev_selector = land.ev_selectors[i]
                        ev_selector.region_label = region.label
                        ev_selector.initial_state = land.state
                        ev_selector.final_state = v
                        ev_selector.ev_labels = ev_labels
    
    def get_ev_selector(self,
                        region_info,
                        land_state,
                        final_state):
        region = self.get_region(region_info)
        land = region.get_land(land_state)
        
        return land.ev_selectors[land.final_states.index(final_state)]
        
    
    # def get_ev_selectors_fit_arguments(self,
    #                                    initial_luc_layer,
    #                                    final_luc_layer,
    #                                    evs,
    #                                    regions_layer=None,
    #                                    unfitted_only=False):
        
    #     initial_luc_layer, final_luc_layer, evs, regions_layer = \
    #         self._layers_pretreatment(initial_luc_layer=initial_luc_layer,
    #                                   final_luc_layer=final_luc_layer,
    #                                   evs=evs,
    #                                   regions_layer=regions_layer)
        
    #     self.init_ev_selectors(evs=evs)
        
    #     bounds = get_bounds(evs=evs)
        
    #     R = []
        
    #     for region in self.regions:
    #         for land in region.lands:
    #             J, W, Z = self._get_J_W_Z(initial_luc_layer=initial_luc_layer,
    #                                      final_luc_layer=final_luc_layer,
    #                                      evs=evs,
    #                                      regions_layer=regions_layer,
    #                                      region=region,
    #                                      land=land)
                
    #             for i, v in enumerate(land.final_states):
    #                 transited_pixels = V == v
    #                 ev_selector = land.ev_selectors[i]
                    
    #                 R.append([ev_selector, Z, transited_pixels, bounds])
                    
        
    #     return R
    
    
    
    def calibrate(self,
                  initial_luc_layer,
                  final_luc_layer,
                  evs,
                  regions_layer=None,
                  region_only=None,
                  land_only=None):
        
        # Layers IO
        initial_luc_layer = open_layer(path=initial_luc_layer, kind='land_use')
        final_luc_layer = open_layer(path=final_luc_layer, kind='land_use')
        regions_layer = open_layer(path=regions_layer, kind='regions')
        for k, ev in enumerate(evs):
            if type(ev) is str:
                evs[k] = open_layer(path=ev, kind='ev')
        
        if self.verbose > 0:
            print("=================")
            print("|| CALIBRATION ||")
            print("=================\n")
        
        regions_list = self.regions
        if region_only is not None:
            regions_list = [self.get_region(region_only)]
        
        bounds = get_bounds(evs=evs)
        
        self.init_ev_selectors(evs=evs)
        
        for region in regions_list:
            
            if self.verbose > 0:
                print("Region : "+region.label)
                s = ''
                for i in range(len(region.label)):
                    s += '='
                print('========='+s+'\n')
            
            
            lands_list = region.lands
            if land_only is not None:
                lands_list = [region.get_land(land_only)]
            
            for land in lands_list:
                
                if self.verbose > 0:
                    print('land : '+str(land.state)+' - '+ self.palette.get(land.state).label)
                    print('---------\n')
                    
            
                # data
                J, W, Z = get_J_W_Z(initial_luc_layer=initial_luc_layer,
                                    final_luc_layer=final_luc_layer,
                                    evs=evs,
                                    regions_layer=regions_layer,
                                    region_value=region.value,
                                    initial_state=land.state,
                                    final_states=land.final_states)
                
                if self.verbose > 0:
                    print('n pixels : '+str("{:.2e}".format(J.size)))
                    print('final states : ', land.final_states)
                
                # Explanatory Variable Selection
                land._ev_selectors = EVSelectors(selectors=land.ev_selectors)
                land._ev_selectors.fit(W, Z, bounds)
                Z = land._ev_selectors.transform(Z)
                
                # TPE
                tpe = land.transition_probability_estimator
                selected_evs = land._ev_selectors.get_selected_evs(evs)
                selected_bounds = get_bounds(selected_evs)
                if self.verbose > 0:
                    print()
                tpe.fit(Z, W, bounds=selected_bounds)
                            
            if self.verbose > 0:
                print()
        
    def _get_P_v_region(self, P_v, region):
        if type(P_v) is not dict:
            return P_v
        
        if region.label in P_v.keys():
            return P_v[region.label]
        
        if region.value in P_v.keys():
            return P_v[region.value]
                
    def _get_P_v_land(self, P_v_region, land):
        list_u = list(P_v_region[:,0])
        list_v = list(P_v_region[0,:])
        
        P_v_land = np.zeros(len(land.final_states))
        
        u = land.state
        for i, v in enumerate(land.final_states):
            P_v_land[i] = P_v_region[list_u.index(u), list_v.index(v)]
        
        return P_v_land
        
    def transition_probabilities(self,
                                 luc_layer,
                                 evs,
                                 P_v,
                                 regions_layer=None,
                                 region_only=None,
                                 land_only=None,
                                 return_initial_state=False):
        """
        Compute the transition probabilities of the case.
        """
        
        # Layers IO
        luc_layer = open_layer(path=luc_layer, kind='land_use')
        regions_layer = open_layer(path=regions_layer, kind='regions')
        for k, ev in enumerate(evs):
            if type(ev) is str:
                evs[k] = open_layer(path=ev, kind='ev')
        
        if self.verbose > 0:
            print("=========================================")
            print("|| TRANSITION PROBABILITIES ESTIMATION ||")
            print("=========================================\n")
        
        regions_list = self.regions
        if region_only is not None:
            regions_list = [self.get_region(region_only)]
        
        P_v__u_Z_layer = None
        
        for region in regions_list:
            
            if self.verbose > 0:
                print("Region : "+region.label)
                s = ''
                for i in range(len(region.label)):
                    s += '='
                print('========='+s+'\n')
            
            
            P_v_region = self._get_P_v_region(P_v, region)
            
            lands_list = region.lands
            if land_only is not None:
                lands_list = [region.get_land(land_only)]
            
            for land in lands_list:
                
                if self.verbose > 0:
                    print('land : '+str(land.state)+' - '+ self.palette.get(land.state).label)
                    print('---------\n')
                
                # data
                J, Z = get_J_Z(luc_layer=luc_layer,
                               evs=evs,
                               regions_layer=regions_layer,
                               region_value=region.value,
                               state=land.state)
                
                # Explanatory Variable Selection
                Z = land._ev_selectors.transform(Z)
                
                # P_v land
                P_v_land = self._get_P_v_land(P_v_region, land)
                
                if self.verbose > 0:
                    print('n pixels : '+str("{:.2e}".format(J.size)))
                    print('final states : ', land.final_states)
                    print('P_v : ', P_v_land)
                    
                # TPE COMPUTING
                tpe = land.transition_probability_estimator
                P_v__u_Z = tpe.compute(Y=Z,
                                       P_v=P_v_land)        
                
                all_states = land.final_states
                if return_initial_state:
                    # adding initial state to probabilities
                    P_v__u_Z = np.hstack(((1-P_v__u_Z.sum(axis=1))[:,None], P_v__u_Z))
                    all_states = [land.state] + land.final_states
                
                
                # create Proba layer object
                P_v__u_Z_layer_land = create_proba_layer(J=J,
                                                         P=P_v__u_Z,
                                                         final_states=all_states,
                                                         shape=luc_layer.shape,
                                                         geo_metadata=luc_layer.geo_metadata)
                
                # fusion
                if P_v__u_Z_layer is None:
                    P_v__u_Z_layer = P_v__u_Z_layer_land
                else:
                    P_v__u_Z_layer = P_v__u_Z_layer.fusion(P_v__u_Z_layer_land)
                
                if self.verbose > 0:
                    print('')
        
        return P_v__u_Z_layer
    
    def allocate(self,
                 luc_layer,
                 evs,
                 P_v,
                 regions_layer=None,
                 region_only=None,
                 land_only=None):
        
        # Layers IO
        luc_layer = open_layer(path=luc_layer, kind='land_use')
        regions_layer = open_layer(path=regions_layer, kind='regions')
        for k, ev in enumerate(evs):
            if type(ev) is str:
                evs[k] = open_layer(path=ev, kind='ev')
        
        if self.verbose > 0:
            print("==============")
            print("|| ALLOCATE ||")
            print("==============\n")
        
        # initialize luc_layer
        luc_layer = luc_layer.copy()
        luc_layer_origin = luc_layer.copy()
        
        regions_list = self.regions
        if region_only is not None:
            regions_list = [self.get_region(region_only)]
                
        for region in regions_list:
            
            if self.verbose > 0:
                print("Region : "+region.label)
                s = ''
                for i in range(len(region.label)):
                    s += '='
                print('========='+s+'\n')
            
            
            P_v_region = self._get_P_v_region(P_v, region)
            
            lands_list = region.lands
            if land_only is not None:
                lands_list = [region.get_land(land_only)]
            
            for land in lands_list:
                
                if self.verbose > 0:
                    print('land : '+str(land.state)+' - '+ self.palette.get(land.state).label)
                    print('---------\n')
                
                # data
                J, Z = get_J_Z(luc_layer=luc_layer,
                               evs=evs,
                               regions_layer=regions_layer,
                               region_value=region.value,
                               state=land.state)
                
                # Explanatory Variable Selection
                Z = land._ev_selectors.transform(Z)
                
                # P_v land
                P_v_land = self._get_P_v_land(P_v_region, land)
                
                if self.verbose > 0:
                    print('n pixels : '+str("{:.2e}".format(J.size)))
                    print('final states : ', land.final_states)
                    print('P_v : ', P_v_land)
                    
                # ALLOCATION
                allocator = land.allocator
                tpe = land.transition_probability_estimator
                luc_layer = allocator.allocate(luc_layer=luc_layer,
                                               Z=Z,
                                               tpe_func=tpe.compute,
                                               regions_layer=regions_layer,
                                               luc_layer_origin=luc_layer_origin)
                # P_v__u_Z = tpe.compute(J=J,
                                       # Y=Z,
                                       # P_v=P_v_land)           
                
                
                if self.verbose > 0:
                    print('')
        
                return land.allocator
                
        
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
        
    # 
        
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