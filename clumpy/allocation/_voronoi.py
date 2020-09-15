from ._allocation import _Allocation, compute_P_vf__vi_from_transition_probability_maps
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
        
        # faudrait penser à virer vf de J_proba
        
        for vi in J_proba.v.i.unique():
            J_vi = J_proba.loc[J_proba.v.i==vi].copy()
            J_vi_seeds = J_vi.sample(frac=1/self.params[vi]['S'], replace=False)
            
            J_vi_tesselated = tesselation.voronoi(J_vi[[]],
                                                J_vi_seeds,
                                                map_shape)
            
            P_z__vi = compute_P_z__vi(J_vi)
            
            P_z__vi_new = compute_P_z__vi(J_vi_seeds, name='P_z__vi_new')
            
            J_vi_seeds.reset_index(drop=False, inplace=True)
            J_vi_seeds.loc[J_vi_seeds.index.values, ('v','i')] = J_vi_seeds.v.i.astype(int)
            J_vi_seeds = J_vi_seeds.merge(P_z__vi, how='left')
            J_vi_seeds = J_vi_seeds.merge(P_z__vi_new, how='left')
            J_vi_seeds.set_index('index', inplace=True)
            
            for vf in J_vi_seeds.P_vf__vi_z.columns.to_list():
                J_vi_seeds[('P_vf__vi_z', vf)] = J_vi_seeds[('P_vf__vi_z', vf)] * J_vi_seeds.P_z__vi_new/ J_vi_seeds.P_z__vi
                # / self.params[vi]['S']
            
            J_vi_seeds.drop(['P_z__vi_new', 'P_z__vi'], axis=1, level=0, inplace=True)
            
            self._generalized_acceptation_rejection_test(J_vi_seeds, inplace=True, accepted_only=True)
            
            S = J_vi_tesselated.groupby('voronoi_seed').size().reset_index(name='S')
            S.set_index('voronoi_seed', inplace=True)
            J_vi_seeds['S'] = S.loc[J_vi_seeds.index.values, 'S'].values
            
            print(J_vi_seeds.S.mean())
            # seeds allocation
            map_f_data.flat[J_vi_seeds.index.values] = J_vi_seeds.v.f.values
            
            # patch allocation
            map_f_data.flat[J_vi_tesselated.index.values] = map_f_data.flat[J_vi_tesselated.voronoi_seed.values]
    
    
        map_f = LandUseCoverLayer(name="luc_simple")
        map_f.import_numpy(data=map_f_data)
        
        return(map_f)
            
            
        
        # for vi in J_proba.v.i.unique():
        #     J_vi = J_proba.loc[J_proba.v.i == vi]
        #     J_vi_seeds = J_vi.sample(frac=1/self.params[vi]['S'],
        #                                                        replace=False)
        
        #     points_x, points_y = np.unravel_index(J_vi_seeds.index.values, map_i.data.shape)
        #     points_seeds = np.zeros((points_x.size,2))
        #     points_seeds[:,0] = points_x
        #     points_seeds[:,1]  = points_y
            
        #     points_x_all, points_y_all = np.unravel_index(J_vi.index.values, map_i.data.shape)
        #     points_all = np.zeros((points_x_all.size,2))
        #     points_all[:,0] = points_x_all
        #     points_all[:,1]  = points_y_all
            
        #     voronoi_kdtree = cKDTree(points_seeds)
        #     test_point_dist, test_point_regions = voronoi_kdtree.query(points_all, k=1)
            
        #     J_vi[('voronoi_seed', '')] = J_vi_seeds.index[test_point_regions]
            
        #     # probability update
        #     # i.e. P(z|vi) update
        #     # and P(vf|vi) update
            
            
        #     P_z__vi = compute_P_z__vi(J_vi)
        #     new_P_z__vi = compute_P_z__vi(J_vi_seeds, name='new_P_z__vi')
            
        #     # P_vf__vi = compute_P_vf__vi_from_P_vf__vi_z(J_vi)
            
                        
        #     # J_vi_seeds_updated = J_vi_seeds.merge(right=P_vf__vi, how='left')
        #     J_vi_seeds.reset_index(drop=False, inplace=True)
        #     J_vi_seeds.loc[J_vi_seeds.index.values, ('v','i')] = J_vi_seeds.v.i.astype(int)
            
        #     J_vi_seeds_updated = J_vi_seeds.merge(right=P_z__vi.astype(float), how='left')
        #     J_vi_seeds_updated = J_vi_seeds_updated.merge(right=new_P_z__vi.astype(float), how='left')
            
        #     for vf in J_vi_seeds_updated.P_vf__vi_z.columns.to_list():
        #         J_vi_seeds_updated[('P_vf__vi_z', vf)] *= J_vi_seeds_updated.new_P_z__vi / J_vi_seeds_updated.P_z__vi / self.params[vi]['S']
                
        #     J_vi_seeds_updated.drop(['P_z__vi', 'new_P_z__vi'], axis=1, level=0, inplace=True)
            
        #     J_vi_seeds_accepted = self._generalized_acceptation_rejection_test(J_vi_seeds_updated, accepted_only=True)
            
        #     J_vi_seeds_accepted.set_index('index', inplace=True)
            
        #     for idx in J_vi_seeds_accepted.index.values:
        #         J_vi.loc[J_vi.voronoi_seed == idx, ('v', 'f')] = J_vi_seeds_accepted.loc[idx].v.f
                
        #     map_f_data.flat[J_vi.index.values] = J_vi.v.f.values
        
        # map_f = LandUseCoverLayer(name="luc_simple",
        #                                        time=None,
        #                                        scale=map_i.scale)
        # map_f.import_numpy(data=map_f_data)
        
        # return(map_f)
    
    
    
