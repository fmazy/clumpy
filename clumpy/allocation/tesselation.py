# from ..definition import get_pixels_coordinates

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial import Voronoi as scipyVoronoi
from scipy import sparse
from tqdm import tqdm

import networkx as nx

def voronoi(J, J_germs, map_shape, name='voronoi_germ', get_adjacent_cells=False):
    
    J = J.copy()
    
    coor_seeds = get_pixels_coordinates(J_germs, map_shape)
    coor = get_pixels_coordinates(J, map_shape)
    
    voronoi_kdtree = cKDTree(coor_seeds)
    test_point_dist, test_point_regions = voronoi_kdtree.query(coor, k=1)
    
    J[name] = J_germs.index.values[test_point_regions]
    
    
    if get_adjacent_cells:
        vor = scipyVoronoi(coor_seeds)
        
        edgelist = list(vor.ridge_dict.keys())
        G = nx.Graph(edgelist)
        
        return(J, G)
    else:
        return(J)

    
def get_colors(G):
    color_map = nx.coloring.greedy_color(G, strategy="largest_first")
    return color_map