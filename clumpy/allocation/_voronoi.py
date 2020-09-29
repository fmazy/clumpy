from ._allocation import _Allocation, compute_P_vf__vi_from_transition_probability_maps, update_P_vf__vi_z
from ..calibration._calibration import _Calibration, compute_P_z__vi, compute_P_vf__vi_from_P_vf__vi_z
from ..definition import LandUseCoverLayer
from ..definition import get_pixels_coordinates
from ._patcher import _weighted_neighbors
from . import tesselation

import numpy as np
import pandas as pd
import time
from scipy.spatial import cKDTree
from scipy.spatial import Voronoi as scipyVoronoi


class Voronoi(_Allocation):
    """Voronoi Allocation Method
    
    Parameters
    ----------
    params : dict (default=None)
        the parameters dictionary
    """
    
    def __init__(self, params = None):
        super().__init__(params)

    def allocate(self,
                 map_i,
                 J_proba):
        map_f_data = map_i.data.copy()
        map_shape = map_i.data.shape
        
        J_proba = J_proba.rename({'P_vf__vi_z':'P_vf__vi_z_original'}, axis=1, level=0)
        
        # vi type guarantee
        J_proba.loc[J_proba.index.values, ('v','i')] = J_proba.v.i.astype(int)
        
        for vi in J_proba.v.i.unique():
            J_vi = J_proba.loc[J_proba.v.i==vi]
            
            # computes P(z|vi) function of J_vi
            P_z__vi_original = compute_P_z__vi(J_vi, name='P_z__vi_original')
            
            # drawn a 1/sigma_0 frac among J_vi to get voronoï germs
            J_vi_germs = J_vi.sample(frac=1/self.params['sigma_0'][vi], replace=False)
            J_vi_germs['germ_id'] = np.arange(J_vi_germs.index.size)
            
            # voronoï diagrams computing
            # G is the networkx graph of the voronoï diagrams.
            # nodes id are the J_vi_germs order, ie J_vi_germs['germ_id']
            J_vi_tesselated, G = tesselation.voronoi(J = J_vi[[]],
                                                J_germs = J_vi_germs,
                                                map_shape = map_shape,
                                                name = 'voronoi_germ',
                                                get_adjacent_cells = True)
            
            # computes P(z|vi) function of J_vi_germs
            P_z__vi_new = compute_P_z__vi(J_vi_germs, name='P_z__vi_new')
            
            # prepare J_vi_germs for updating
            J_vi_germs.reset_index(drop=False, inplace=True)
                        
            # updates P_vf__vi_z inside J_vi_germs
            J_vi_germs = update_P_vf__vi_z(P_vf__vi_z_original=J_vi_germs,
                              P_z__vi_original = P_z__vi_original,
                              P_z__vi_new = P_z__vi_new)
                        
            # replace index inside J_vi_germs
            J_vi_germs.set_index('index', inplace=True)
            
            # initialize counters
            n_vi_vf_allocated = {}
            n_vi_vf_avorted = {}
            
            for vf in J_vi_germs.P_vf__vi_z.columns.to_list():
                n_vi_vf_allocated[vf] = 0
                n_vi_vf_avorted[vf] = 1
                
                
            
            # GART
            J_vi_c = self._generalized_acceptation_rejection_test(J_vi_germs, inplace=False, accepted_only=True)
            
            # allocation germs
            map_f_data.flat[J_vi_c.index.values] = J_vi_c.v.f.values
            
            # patch allocation
            map_f_data.flat[J_vi_tesselated.index.values] = map_f_data.flat[J_vi_tesselated.voronoi_germ.values]
        
        map_f = LandUseCoverLayer(name="luc_simple")
        map_f.import_numpy(data=map_f_data)
        
        return(map_f)

    def allocate_mono_voronoi_cell(self,
                 map_i,
                 J_proba):
        map_f_data = map_i.data.copy()
        map_shape = map_i.data.shape
        
        J_proba = J_proba.rename({'P_vf__vi_z':'P_vf__vi_z_original'}, axis=1, level=0)
        
        # vi type guarantee
        J_proba.loc[J_proba.index.values, ('v','i')] = J_proba.v.i.astype(int)
        
        for vi in J_proba.v.i.unique():
            J_vi = J_proba.loc[J_proba.v.i==vi]
            
            # computes P(z|vi) function of J_vi
            P_z__vi_original = compute_P_z__vi(J_vi, name='P_z__vi_original')
            
            # drawn a 1/sigma_0 frac among J_vi to get voronoï germs
            J_vi_germs = J_vi.sample(frac=1/self.params['sigma_0'][vi], replace=False)
            J_vi_germs['germ_id'] = np.arange(J_vi_germs.index.size)
            
            # voronoï diagrams computing
            # G is the networkx graph of the voronoï diagrams.
            # nodes id are the J_vi_germs order, ie J_vi_germs['germ_id']
            J_vi_tesselated, G = tesselation.voronoi(J = J_vi[[]],
                                                J_germs = J_vi_germs,
                                                map_shape = map_shape,
                                                name = 'voronoi_germ',
                                                get_adjacent_cells = True)
            
            # computes P(z|vi) function of J_vi_germs
            P_z__vi_new = compute_P_z__vi(J_vi_germs, name='P_z__vi_new')
            
            # prepare J_vi_germs for updating
            J_vi_germs.reset_index(drop=False, inplace=True)
                        
            # updates P_vf__vi_z inside J_vi_germs
            J_vi_germs = update_P_vf__vi_z(P_vf__vi_z_original=J_vi_germs,
                              P_z__vi_original = P_z__vi_original,
                              P_z__vi_new = P_z__vi_new)
                        
            # replace index inside J_vi_germs
            J_vi_germs.set_index('index', inplace=True)
            
            # GART
            J_vi_c = self._generalized_acceptation_rejection_test(J_vi_germs, inplace=False, accepted_only=True)
            
            # allocation germs
            map_f_data.flat[J_vi_c.index.values] = J_vi_c.v.f.values
            
            # patch allocation
            map_f_data.flat[J_vi_tesselated.index.values] = map_f_data.flat[J_vi_tesselated.voronoi_germ.values]
        
        map_f = LandUseCoverLayer(name="luc_simple")
        map_f.import_numpy(data=map_f_data)
        
        return(map_f)
    
    
