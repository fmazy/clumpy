"""
intro
"""

from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy import ndimage # used for distance computing and patch measurements
from matplotlib import pyplot as plt
from copy import deepcopy
from skimage import measure # for patch perimeters
from ...definition import make_J
from ...tools import np_drop_duplicates_from_column


def analyse(case,
            initial_luc_layer,
            final_luc_layer,
            neighbors_structure='queen'):
    
    if neighbors_structure == 'queen':
        structure = np.ones((3,3))
    elif neighbors_structure == 'rook':
        structure = np.array([[0,1,0],
                              [1,1,1],
                              [0,1,0]])
    else:
        raise(ValueError('ERROR : unexpected neighbors_structure value'))
    
    M_shape = initial_luc_layer.get_data().shape
    
    patches = {}
    
    for u in case.params.keys():
        patches[u] = {}
        
        J_u, v_u = case.make(initial_luc_layer = initial_luc_layer,
                                 final_luc_layer = final_luc_layer,
                                 explanatory_variables = False)
        
        for v in case.params[u]['v']:
            if u != v:
                patches[u][v] = {}
                
                print(str(u)+' -> '+str(v))
                
                M = np.zeros(M_shape)
                M.flat[J_u[u][v_u[u] == v]] = 1
                
                lw, _ = ndimage.measurements.label(M, structure=structure)
                patch_id = lw.flat[J_u[u]]
                
                # unique pixel for a patch
                one_pixel_from_patch = np.column_stack((J_u[u], patch_id))
                one_pixel_from_patch = np_drop_duplicates_from_column(one_pixel_from_patch, 1)
                
                one_pixel_from_patch = one_pixel_from_patch[1:, :]
                one_pixel_from_patch[:,1] -= 1
                
                patches[u][v]['J'] = one_pixel_from_patch[:,0]
                patches[u][v]['patch_id'] = one_pixel_from_patch[:,1]
                
                rpt = measure.regionprops_table(lw, properties=['area',
                                                                'inertia_tensor_eigvals'])
                            
                patches[u][v]['area'] = np.array(rpt['area'])
                
                # return(patches, rpt)
                l1_patch = np.array(rpt['inertia_tensor_eigvals-0'])[patches[u][v]['patch_id']]
                l2_patch = np.array(rpt['inertia_tensor_eigvals-1'])[patches[u][v]['patch_id']]
                
                patches[u][v]['eccentricity'] = np.zeros(patches[u][v]['area'].shape)
                id_none_mono_pixel_patches = patches[u][v]['area'] > 1
                
                patches[u][v]['eccentricity'][id_none_mono_pixel_patches] = 1 - np.sqrt(l2_patch[id_none_mono_pixel_patches] / l1_patch[id_none_mono_pixel_patches])
            
    return(patches)
        
def analyse_isl_exp(case, initial_luc_layer, final_luc_layer, neighbors_structure='queen'):
    
    if neighbors_structure == 'queen':
        structure = np.ones((3,3))
    elif neighbors_structure == 'rook':
        structure = np.array([[0,1,0],
                              [1,1,1],
                              [0,1,0]])
    else:
        raise(ValueError('ERROR : unexpected neighbors_structure value'))
    
    M_shape = initial_luc_layer.get_data().shape
    
    list_unique_v = []
    for params in case.params.values():
        for v in params['v']:
            if v not in list_unique_v:
                list_unique_v.append(v)
    
    J_v = {}
    patch_id_v = {}
    area_v = {}
    
    for v in list_unique_v:
        print('v=', v)
        # final pixels
        # the region is not taken into account here.
        J_v[v] = make_J(initial_luc_layer = final_luc_layer,
                          u = v)
        
        # map of final pixels
        M_v = np.zeros(M_shape)
        M_v.flat[J_v[v]] = 1
        
        # patch id of v
        lw_v, _ = ndimage.measurements.label(M_v, structure=structure)
        patch_id_v[v] = lw_v.flat[J_v[v]]
        
        print('\tnumber of patches:', patch_id_v[v].max())
        
        # area of v patches
        area_v[v] = np.array(measure.regionprops_table(lw_v, properties=['area'])['area'])
    
    print('----')
    
    patches = {}
    for u in case.params.keys():
        patches[u] = {}
        for v in case.params[u]['v']:
            print(str(u)+' -> '+str(v))
            
            # transited pixels among J_v
            # the region is taken into account here.
            J_u_v = make_J(initial_luc_layer = initial_luc_layer,
                             u = u,
                             final_luc_layer= final_luc_layer,
                             v = v,
                             region = case.region)
            
            if J_u_v.size > 0:
                patches[u][v] = {}
                # map of transited pixels
                M_u_v = np.zeros(M_shape)
                M_u_v.flat[J_u_v] = 1
                
                # patch id of u->v
                lw_u_v, _ = ndimage.measurements.label(M_u_v, structure=structure)
                patch_id_u_v = lw_u_v.flat[J_u_v]
                
                # unique pixel for a patch
                one_pixel_from_patch_u_v = np.column_stack((J_u_v, patch_id_u_v))
                one_pixel_from_patch_u_v = np_drop_duplicates_from_column(one_pixel_from_patch_u_v, 1)
                
                patches[u][v]['J'] = one_pixel_from_patch_u_v[:,0]
                patches[u][v]['patch_id'] = one_pixel_from_patch_u_v[:,1]
                
                # area of u v patches
                rpt = measure.regionprops_table(lw_u_v, properties=['area',
                                                                      'inertia_tensor_eigvals'])
                
                patches[u][v]['area'] = np.array(rpt['area'])[one_pixel_from_patch_u_v[:,1]-1]
                
                l1_patch_u_v = np.array(rpt['inertia_tensor_eigvals-0'])[patches[u][v]['patch_id']-1]
                l2_patch_u_v = np.array(rpt['inertia_tensor_eigvals-1'])[patches[u][v]['patch_id']-1]
                
                patches[u][v]['eccentricity'] = np.zeros(patches[u][v]['area'].shape)
                id_none_mono_pixel_patches = patches[u][v]['area'] > 1
                
                patches[u][v]['eccentricity'][id_none_mono_pixel_patches] = 1 - np.sqrt(l2_patch_u_v[id_none_mono_pixel_patches] / l1_patch_u_v[id_none_mono_pixel_patches])
                            
                corresponding_area_v = area_v[v][patch_id_v[v][np.searchsorted(J_v[v],
                                                                                    patches[u][v]['J'])]-1]
                
                patches[u][v]['island'] = corresponding_area_v == patches[u][v]['area']
            
    return(patches)

def compute_isl_ratio(patches):
    r = {}
    for u in patches.keys():
        r[u] = {}
        for v in patches[u].keys():
            print(u,v)
            r[u][v] = (patches[u][v]['island']*patches[u][v]['area']).sum()/patches[u][v]['area'].sum()
            # r[u].append((patches[u][v]['island']*patches[u][v]['area']).sum()/patches[u][v]['area'].sum())
        # r[u] = np.array(r[u])
            
    return(r)

def remove_to_big_areas(patches, vi, vf, isl_exp, m, inplace=False):
    if not inplace:
        patches = deepcopy(patches)
    
    id_isl_exp = patches[vi][vf]['island']
    if isl_exp == 'exp':
        id_isl_exp = ~id_isl_exp
        
    id_patches_to_keep = ~((patches[vi][vf]['area'] > m) & id_isl_exp)
    # print(str(round((~id_patches_to_keep).mean()*100,4))+'% removed')
    
    for p in ['J', 'patch_id', 'island', 'area', 'eccentricity']:
        patches[vi][vf][p] = patches[vi][vf][p][id_patches_to_keep]
    
    if not inplace:
        return(patches)

def compute_histograms(patches, name, bins=None, plot=True):
    h = {}

    for vi in patches.keys():
        h[vi] = {}
        for vf in patches[vi].keys():
            h[vi][vf] = {}
            
            for isl_exp in ['isl', 'exp']:
                
                b = 'auto'
                if bins is not None:
                    b = bins[vi][vf][isl_exp]
                
                if isl_exp == 'isl':
                    idx = patches[vi][vf]['island']
                else:
                    idx = ~patches[vi][vf]['island']
                
                N, S = np.histogram(patches[vi][vf][name][idx],
                                                    bins=b,
                                                    density=False)
                
                h[vi][vf][isl_exp] = [N/N.sum(), S]
            
                if plot:
                    plot_histogram(h[vi][vf][isl_exp],
                                   t='bar')
                    plt.xlabel(name)
                    plt.ylabel('P')
                    plt.title(str(vi)+'->'+str(vf)+' - '+str(isl_exp))
                    plt.show()
                
    return(h)


def plot_histogram(h, t='step', color=None, linestyle=None, linewidth=None, label=None):

    if t == 'step':
        y = np.append(h[0], 0)
        x = h[1]
        plt.step(x=x,
                y=y,
                where='post',
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label)
    
    elif t =='bar':
        height = h[0]
        x = h[1][:-1]
        
        plt.bar(x=x,
                height=height,
                width=np.diff(h[1])*0.9,
                align='edge',
                color=color,
                label=label)



# def plot_surfaces_histogram(surfaces_histogram, vi, vf, isl_exp, color=None, linestyle=None, linewidth=None, label=None):
#     part = surfaces_histogram.loc[(surfaces_histogram.vi == vi) &
#                                (surfaces_histogram.vf == vf) &
#                                (surfaces_histogram.isl_exp == isl_exp)]
    
#     S = part.S.values
#     N = part.N.values
    
#     plt.step(x=S,
#             y=N,
#             where='post',
#             color=color,
#             linestyle=linestyle,
#             linewidth=linewidth,
#             label=label)

# def generateBigNp(shape, index, values, default_value=0):
#     M = np.zeros(shape)
#     M.fill(default_value)
#     M.flat[index] = values
#     return(M)


# def analyseDelta(Tif):
#     df_J_vi_vf = Tif.Ti.J_vi.loc[Tif.Ti.J_vi.vf == Tif.vf].copy()
#     map_vi_vf = generateBigNp(np.shape(Tif.Ti.T.map_i.data),
#                                       df_J_vi_vf.index.values,
#                                       1)   
    
#     convolution_mask = np.zeros((3,3))
#     convolution_mask.flat[[1,3,5,7]] = 1
#     # calcul du nombre de voisins
#     map_vf_convoluted = signal.convolve2d(map_vi_vf,convolution_mask,'same')
    
#     maps_vi_vf_neighbors = map_vf_convoluted * map_vi_vf
# #    print('a',len(np.where(maps_vi_vf_neighbors==4)))
#     maps_vi_vf_neighbors[maps_vi_vf_neighbors==4] = 0
#     maps_vi_vf_neighbors[maps_vi_vf_neighbors!=0] = 1
# #    print('b',maps_vi_vf_neighbors.sum())
    
#     lw_vi_vf, num_vi_vf = ndimage.measurements.label(map_vi_vf) # création des îlots
# #    
#     maps_vi_vf_neighbors_lw_vi_vf = maps_vi_vf_neighbors*lw_vi_vf
    
#     delta = np.array([])
    
#     for n_vi_vf in tqdm(range(1,num_vi_vf+1)):
#         j_n_vi_vf = np.where(lw_vi_vf.flat==n_vi_vf)[0]
#         # c'est quelles coordonnées?
#         coord = np.unravel_index(j_n_vi_vf, np.shape(Tif.Ti.T.map_i.data))
#         G = [np.mean(coord[0]), np.mean(coord[1])]
        
#         j_n_vi_vf_side = np.where(maps_vi_vf_neighbors_lw_vi_vf.flat==n_vi_vf)[0]
        
#         coord = np.unravel_index(j_n_vi_vf_side, np.shape(Tif.Ti.T.map_i.data))
#         d = np.sqrt(np.power(coord[0]-G[0],2)+np.power(coord[1]-G[1],2))
        
#             # if all distances are equals, they are normalized to 0.5
#             # (very unlike in a real case...)
#         if len(d)>0:
#             if np.mean(d) == np.max(d):
#                 d.fill(0.5)
#             else: 
#                 # else, they are reduced to [0,1]
#                 d = d-np.min(d)
#                 d = d/np.max(d)
                
#             delta = np.append(delta, d)
        
#     return(delta)

# def analyseDelta2(Tif):
#     df_J_vi_vf = Tif.Ti.J_vi.loc[Tif.Ti.J_vi.vf == Tif.vf].copy()
#     map_vi_vf = dmtools.generateBigNp(np.shape(Tif.Ti.T.map_i.data),
#                                       df_J_vi_vf.index.values,
#                                       1)   
    
#     convolution_mask = np.zeros((3,3))
#     convolution_mask.flat[[1,3,5,7]] = 1
#     # calcul du nombre de voisins
#     map_vf_convoluted = signal.convolve2d(map_vi_vf,convolution_mask,'same')
    
#     maps_vi_vf_neighbors = map_vf_convoluted * map_vi_vf
# #    print('a',len(np.where(maps_vi_vf_neighbors==4)))
#     maps_vi_vf_neighbors[maps_vi_vf_neighbors==4] = 0
#     maps_vi_vf_neighbors[maps_vi_vf_neighbors!=0] = 1
# #    print('b',maps_vi_vf_neighbors.sum())
    
#     lw_vi_vf, num_vi_vf = ndimage.measurements.label(map_vi_vf) # création des îlots
# #    
#     maps_vi_vf_neighbors_lw_vi_vf = maps_vi_vf_neighbors*lw_vi_vf
    
#     delta = np.array([])
    
#     for n_vi_vf in tqdm(range(1,num_vi_vf+1)):
#         j_n_vi_vf = np.where(lw_vi_vf.flat==n_vi_vf)[0]
#         # c'est quelles coordonnées?
#         coord = np.unravel_index(j_n_vi_vf, np.shape(Tif.Ti.T.map_i.data))
#         G = [np.mean(coord[0]), np.mean(coord[1])]
        
#         j_n_vi_vf_side = np.where(maps_vi_vf_neighbors_lw_vi_vf.flat==n_vi_vf)[0]
        
#         coord = np.unravel_index(j_n_vi_vf_side, np.shape(Tif.Ti.T.map_i.data))
#         d = np.sqrt(np.power(coord[0]-G[0],2)+np.power(coord[1]-G[1],2))
        

#         if len(d)>0:
#             d = d/np.mean(d)
                
#             delta = np.append(delta, d)
        
#     return(delta)
        
            
# def histogram(Tif, isl_exp):    
#     # d'abord, on supprime les entrées correspondantes dans le dataframe
#     Tif.Ti.T.patchesHist = Tif.Ti.T.patchesHist.loc[~((Tif.Ti.T.patchesHist.vi == Tif.Ti.vi) &
#                                                         (Tif.Ti.T.patchesHist.vf == Tif.vf) &
#                                                         (Tif.Ti.T.patchesHist.isl_exp == isl_exp))]
    
#     patches_N, patches_S = np.histogram(a=Tif.Ti.T.patches.loc[(Tif.Ti.T.patches.vi==Tif.Ti.vi) &
#                                                            (Tif.Ti.T.patches.vf==Tif.vf) &
#                                                            (Tif.Ti.T.patches.isl_exp==isl_exp) &
#                                                            (Tif.Ti.T.patches.S <= Tif.patches_param[isl_exp]['S_max'])].S.values,
#                                         bins=Tif.patches_param[isl_exp]['bins'],
#                                         density=False)
#     patches_N = np.append(patches_N, 0) # faut que les listes aient la même longueur. ça conclut les bins.
    
#     df_patches = pd.DataFrame(np.array([patches_S, patches_N]).T)
#     df_patches["vi"] = Tif.Ti.vi
#     df_patches["vf"] = Tif.vf
#     df_patches["isl_exp"] = isl_exp
#     df_patches.columns = ['S', 'N', 'vi', 'vf', 'isl_exp']
    
#     Tif.Ti.T.patchesHist = pd.concat([Tif.Ti.T.patchesHist, df_patches], ignore_index=True)
    
# def histogramAllTif(T):        
#     for Ti in T.Ti.values():
#         for Tif in Ti.Tif.values():
#             for isl_exp in ['isl', 'exp']:
#                 histogram(Tif, isl_exp=isl_exp)
                        
# def setOffHistogramAsNormal(Tif, islands_or_expansions:str):
#     """
#     Set off the normal distribution of patches.
    
#     :param Tif: :math:`T_{v_i,v_f}`
#     :type Tif: ._transition._Transition_vi_vf
#     :param islands_or_expansions: island or expansion type -- ``"islands"`` or ``"expansions"``.
#     :type islands_or_expansions: string
#     """
#     if islands_or_expansions== 'islands':
#         Tif.patches_islands_normal_distribution = None
#     elif islands_or_expansions== 'expansions':
#         Tif.patches_expansions_normal_distribution = None
                
# def display(Tif, isl_exp, density=True, hectare=True):       
#     patchesHist = Tif.Ti.T.patchesHist.loc[(Tif.Ti.T.patchesHist.vi == Tif.Ti.vi) &
#                                             (Tif.Ti.T.patchesHist.vf == Tif.vf) &
#                                             (Tif.Ti.T.patchesHist.isl_exp == isl_exp)]
    
#     if density:
#         coef = 1/patchesHist.N.sum()
#     else:
#         coef = 1
        
#     if hectare:
#         hectare_coef = Tif.Ti.T.map_i.scale**2/10000
#     else:
#         hectare_coef = 1
    
#     plt.bar(x=patchesHist.S.values[:-1]*hectare_coef,
#             height=patchesHist.N.values[:-1]*coef,
#             width=np.diff(patchesHist.S.values*hectare_coef),
#             align='edge')
# #        plt.step(x=patchesHist.S.values[:]*hectare_coef,
# #                y=patchesHist.N.values[:]*coef,
# ##                width=np.diff(patchesHist.S.values*hectare_coef),
# #                where='post', label=islands_or_expansions)
# #        plt.title(str(Tif)+' '+islands_or_expansions)
#     plt.grid()
#     if hectare:
#         plt.xlabel("ha")
#     else:
#         plt.xlabel("px")
#     plt.ylabel("frequency")
# #        plt.show()
