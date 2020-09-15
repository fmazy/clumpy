from ..definition import get_pixels_coordinates

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial import Voronoi as scipyVoronoi
from scipy import sparse
from tqdm import tqdm

import networkx as nx

def voronoi(J, J_seeds, map_shape, get_adjacent_cells=False):
    
    J = J.copy()
    
    coor_seeds = get_pixels_coordinates(J_seeds, map_shape)
    coor = get_pixels_coordinates(J, map_shape)
    
    voronoi_kdtree = cKDTree(coor_seeds)
    test_point_dist, test_point_regions = voronoi_kdtree.query(coor, k=1)
    
    J[('voronoi_seed')] = J_seeds.index.values[test_point_regions]
    
    
    if get_adjacent_cells:
        vor = scipyVoronoi(coor_seeds)
        
        edgelist = list(vor.ridge_dict.keys())
        G = nx.Graph(edgelist)
        
        # adjacents = sparse.coo_matrix((np.ones(vor.ridge_points.shape[0]), (vor.ridge_points[:,0], vor.ridge_points[:,1])))
        
        # adjacents = [[] for i in range(vor.ridge_points.max().max()+1)]
        # adjacents = {}
        # for i in range(vor.ridge_points.max().max()+1):
        #     adjacents[i] = []
        # for id_ridge in range(vor.ridge_points.shape[0]):
        #     adjacents[vor.ridge_points[id_ridge, 0]].append(vor.ridge_points[id_ridge, 1])
        #     adjacents[vor.ridge_points[id_ridge, 1]].append(vor.ridge_points[id_ridge, 0])
        
        # adjacents = pd.DataFrame(vor.ridge_points, columns=['A', 'B'])
        
        # adjacents_full = pd.DataFrame()
        # adjacents_full['A'] = adjacents.B.values
        # adjacents_full['B'] = adjacents.A.values
        # adjacents_full = pd.concat([adjacents, adjacents_full], sort=False)
        
        # adjacents_list = []
        # for i in tqdm(range(adjacents.max().max())):
        #     adjacents_list.append(adjacents.loc[adjacents.A == i].B.values)
            
        # adjacents['distance'] = np.linalg.norm(coor_seeds[adjacents.A.values]-coor_seeds[adjacents.B.values], axis=1)
        
        # adjacents['A'] = J_seeds.index.values[adjacents['A'].values]
        # adjacents['B'] = J_seeds.index.values[adjacents['B'].values]
        
        return(J, G)
    else:
        return(J)

    
def get_colors(G):
    color_map = nx.coloring.greedy_color(G, strategy="largest_first")
    return color_map