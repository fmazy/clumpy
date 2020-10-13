"""
intro
"""

from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy import ndimage # used for distance computing and patch measurements
from matplotlib import pyplot as plt
from copy import deepcopy
#from PIL import Image
#from scipy import signal, sparse
from skimage import measure # for patch perimeters

from ..definition import _transition
from .. import tools as dmtools
from ..tools import np_suitable_integer_type
from ..tools import np_drop_duplicates_from_column
from ..tools import plot_histogram


def analyse(case, neighbors_structure = 'queen'):

    
    if neighbors_structure == 'queen':
        structure = np.ones((3,3))
    elif neighbors_structure == 'rook':
        structure = np.array([[0,1,0],
                              [1,1,1],
                              [0,1,0]])
    else:
        print('ERROR : unexpected neighbors_structure value')
        return(False)
    
    list_unique_vf = []
    for vi in case.dict_vi_vf.keys():
        for vf in case.dict_vi_vf[vi]:
            if vf not in list_unique_vf:
                list_unique_vf.append(vf)
    
    J_vf = {}
    patch_id_vf = {}
    area_vf = {}
    
    for vf in list_unique_vf:
        print('vf=', vf)
        # final pixels
        J_vf[vf] = np_suitable_integer_type(np.where(case.map_f.data.flat==vf)[0])
        
        
        # map of final pixels
        M_vf = np.zeros(case.map_i.data.shape)
        M_vf.flat[J_vf[vf]] = 1
        
        # patch id of vf
        lw_vf, _ = ndimage.measurements.label(M_vf, structure=structure)
        patch_id_vf[vf] = lw_vf.flat[J_vf[vf]]
        
        print('\t number of patches:', patch_id_vf[vf].max())
        
        # area of vf patches
        area_vf[vf] = np.array(measure.regionprops_table(lw_vf, properties=['area'])['area'])
    
    print('----')
    patches = {}
    for vi in case.dict_vi_vf.keys():
        patches[vi] = {}
        for vf in case.dict_vi_vf[vi]:
            print(str(vi)+' -> '+str(vf))
            
            patches[vi][vf] = {}
            
            # transited pixels among J_vf
            J_vi_vf = case.J[vi][case.vf[vi] == vf]
            
            # map of transited pixels
            M_vi_vf = np.zeros(case.map_i.data.shape)
            M_vi_vf.flat[J_vi_vf] = 1
            
            # patch id of vi->vf
            lw_vi_vf, _ = ndimage.measurements.label(M_vi_vf, structure=structure)
            patch_id_vi_vf = lw_vi_vf.flat[J_vi_vf]
            
            # unique pixel for a patch
            one_pixel_from_patch_vi_vf = np.column_stack((J_vi_vf, patch_id_vi_vf))
            one_pixel_from_patch_vi_vf = np_drop_duplicates_from_column(one_pixel_from_patch_vi_vf, 1)
            
            patches[vi][vf]['J'] = one_pixel_from_patch_vi_vf[:,0]
            patches[vi][vf]['patch_id'] = one_pixel_from_patch_vi_vf[:,1]
            
            # area of vi vf patches
            rpt = measure.regionprops_table(lw_vi_vf, properties=['area',
                                                                  'inertia_tensor_eigvals'])
            
            patches[vi][vf]['area'] = np.array(rpt['area'])[one_pixel_from_patch_vi_vf[:,1]-1]
            
            l1_patch_vi_vf = np.array(rpt['inertia_tensor_eigvals-0'])[patches[vi][vf]['patch_id']-1]
            l2_patch_vi_vf = np.array(rpt['inertia_tensor_eigvals-1'])[patches[vi][vf]['patch_id']-1]
            
            patches[vi][vf]['eccentricity'] = np.zeros(patches[vi][vf]['area'].shape)
            id_none_mono_pixel_patches = patches[vi][vf]['area'] > 1
            
            patches[vi][vf]['eccentricity'][id_none_mono_pixel_patches] = 1 - np.sqrt(l2_patch_vi_vf[id_none_mono_pixel_patches] / l1_patch_vi_vf[id_none_mono_pixel_patches])
                        
            corresponding_area_vf = area_vf[vf][patch_id_vf[vf][np.searchsorted(J_vf[vf],
                                                                                patches[vi][vf]['J'])]-1]
            
            patches[vi][vf]['island'] = corresponding_area_vf == patches[vi][vf]['area']
            
    return(patches)

def remove_to_big_areas(patches, vi, vf, isl_exp, m, inplace=False):
    if not inplace:
        patches = deepcopy(patches)
    
    id_isl_exp = patches[vi][vf]['island']
    if isl_exp == 'exp':
        id_isl_exp = ~id_isl_exp
        
    id_patches_to_keep = ~((patches[vi][vf]['area'] > m) & id_isl_exp)
    print(id_patches_to_keep.mean())
    
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
                                   t='bar',
                                   title=str(vi)+'->'+str(vf)+' - '+str(isl_exp),
                                   show=True)
                
    return(h)


    


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

def analyseDelta(Tif):
    df_J_vi_vf = Tif.Ti.J_vi.loc[Tif.Ti.J_vi.vf == Tif.vf].copy()
    map_vi_vf = dmtools.generateBigNp(np.shape(Tif.Ti.T.map_i.data),
                                      df_J_vi_vf.index.values,
                                      1)   
    
    convolution_mask = np.zeros((3,3))
    convolution_mask.flat[[1,3,5,7]] = 1
    # calcul du nombre de voisins
    map_vf_convoluted = signal.convolve2d(map_vi_vf,convolution_mask,'same')
    
    maps_vi_vf_neighbors = map_vf_convoluted * map_vi_vf
#    print('a',len(np.where(maps_vi_vf_neighbors==4)))
    maps_vi_vf_neighbors[maps_vi_vf_neighbors==4] = 0
    maps_vi_vf_neighbors[maps_vi_vf_neighbors!=0] = 1
#    print('b',maps_vi_vf_neighbors.sum())
    
    lw_vi_vf, num_vi_vf = ndimage.measurements.label(map_vi_vf) # création des îlots
#    
    maps_vi_vf_neighbors_lw_vi_vf = maps_vi_vf_neighbors*lw_vi_vf
    
    delta = np.array([])
    
    for n_vi_vf in tqdm(range(1,num_vi_vf+1)):
        j_n_vi_vf = np.where(lw_vi_vf.flat==n_vi_vf)[0]
        # c'est quelles coordonnées?
        coord = np.unravel_index(j_n_vi_vf, np.shape(Tif.Ti.T.map_i.data))
        G = [np.mean(coord[0]), np.mean(coord[1])]
        
        j_n_vi_vf_side = np.where(maps_vi_vf_neighbors_lw_vi_vf.flat==n_vi_vf)[0]
        
        coord = np.unravel_index(j_n_vi_vf_side, np.shape(Tif.Ti.T.map_i.data))
        d = np.sqrt(np.power(coord[0]-G[0],2)+np.power(coord[1]-G[1],2))
        
            # if all distances are equals, they are normalized to 0.5
            # (very unlike in a real case...)
        if len(d)>0:
            if np.mean(d) == np.max(d):
                d.fill(0.5)
            else: 
                # else, they are reduced to [0,1]
                d = d-np.min(d)
                d = d/np.max(d)
                
            delta = np.append(delta, d)
        
    return(delta)

def analyseDelta2(Tif):
    df_J_vi_vf = Tif.Ti.J_vi.loc[Tif.Ti.J_vi.vf == Tif.vf].copy()
    map_vi_vf = dmtools.generateBigNp(np.shape(Tif.Ti.T.map_i.data),
                                      df_J_vi_vf.index.values,
                                      1)   
    
    convolution_mask = np.zeros((3,3))
    convolution_mask.flat[[1,3,5,7]] = 1
    # calcul du nombre de voisins
    map_vf_convoluted = signal.convolve2d(map_vi_vf,convolution_mask,'same')
    
    maps_vi_vf_neighbors = map_vf_convoluted * map_vi_vf
#    print('a',len(np.where(maps_vi_vf_neighbors==4)))
    maps_vi_vf_neighbors[maps_vi_vf_neighbors==4] = 0
    maps_vi_vf_neighbors[maps_vi_vf_neighbors!=0] = 1
#    print('b',maps_vi_vf_neighbors.sum())
    
    lw_vi_vf, num_vi_vf = ndimage.measurements.label(map_vi_vf) # création des îlots
#    
    maps_vi_vf_neighbors_lw_vi_vf = maps_vi_vf_neighbors*lw_vi_vf
    
    delta = np.array([])
    
    for n_vi_vf in tqdm(range(1,num_vi_vf+1)):
        j_n_vi_vf = np.where(lw_vi_vf.flat==n_vi_vf)[0]
        # c'est quelles coordonnées?
        coord = np.unravel_index(j_n_vi_vf, np.shape(Tif.Ti.T.map_i.data))
        G = [np.mean(coord[0]), np.mean(coord[1])]
        
        j_n_vi_vf_side = np.where(maps_vi_vf_neighbors_lw_vi_vf.flat==n_vi_vf)[0]
        
        coord = np.unravel_index(j_n_vi_vf_side, np.shape(Tif.Ti.T.map_i.data))
        d = np.sqrt(np.power(coord[0]-G[0],2)+np.power(coord[1]-G[1],2))
        

        if len(d)>0:
            d = d/np.mean(d)
                
            delta = np.append(delta, d)
        
    return(delta)
        
            
def histogram(Tif:_transition._Transition_vi_vf, isl_exp):    
    # d'abord, on supprime les entrées correspondantes dans le dataframe
    Tif.Ti.T.patchesHist = Tif.Ti.T.patchesHist.loc[~((Tif.Ti.T.patchesHist.vi == Tif.Ti.vi) &
                                                        (Tif.Ti.T.patchesHist.vf == Tif.vf) &
                                                        (Tif.Ti.T.patchesHist.isl_exp == isl_exp))]
    
    patches_N, patches_S = np.histogram(a=Tif.Ti.T.patches.loc[(Tif.Ti.T.patches.vi==Tif.Ti.vi) &
                                                           (Tif.Ti.T.patches.vf==Tif.vf) &
                                                           (Tif.Ti.T.patches.isl_exp==isl_exp) &
                                                           (Tif.Ti.T.patches.S <= Tif.patches_param[isl_exp]['S_max'])].S.values,
                                        bins=Tif.patches_param[isl_exp]['bins'],
                                        density=False)
    patches_N = np.append(patches_N, 0) # faut que les listes aient la même longueur. ça conclut les bins.
    
    df_patches = pd.DataFrame(np.array([patches_S, patches_N]).T)
    df_patches["vi"] = Tif.Ti.vi
    df_patches["vf"] = Tif.vf
    df_patches["isl_exp"] = isl_exp
    df_patches.columns = ['S', 'N', 'vi', 'vf', 'isl_exp']
    
    Tif.Ti.T.patchesHist = pd.concat([Tif.Ti.T.patchesHist, df_patches], ignore_index=True)
    
def histogramAllTif(T:_transition._Transition):        
    for Ti in T.Ti.values():
        for Tif in Ti.Tif.values():
            for isl_exp in ['isl', 'exp']:
                histogram(Tif, isl_exp=isl_exp)
                        
def setOffHistogramAsNormal(Tif:_transition._Transition_vi_vf, islands_or_expansions:str):
    """
    Set off the normal distribution of patches.
    
    :param Tif: :math:`T_{v_i,v_f}`
    :type Tif: ._transition._Transition_vi_vf
    :param islands_or_expansions: island or expansion type -- ``"islands"`` or ``"expansions"``.
    :type islands_or_expansions: string
    """
    if islands_or_expansions== 'islands':
        Tif.patches_islands_normal_distribution = None
    elif islands_or_expansions== 'expansions':
        Tif.patches_expansions_normal_distribution = None
                
def display(Tif:_transition._Transition_vi_vf, isl_exp, density=True, hectare=True):       
    patchesHist = Tif.Ti.T.patchesHist.loc[(Tif.Ti.T.patchesHist.vi == Tif.Ti.vi) &
                                            (Tif.Ti.T.patchesHist.vf == Tif.vf) &
                                            (Tif.Ti.T.patchesHist.isl_exp == isl_exp)]
    
    if density:
        coef = 1/patchesHist.N.sum()
    else:
        coef = 1
        
    if hectare:
        hectare_coef = Tif.Ti.T.map_i.scale**2/10000
    else:
        hectare_coef = 1
    
    plt.bar(x=patchesHist.S.values[:-1]*hectare_coef,
            height=patchesHist.N.values[:-1]*coef,
            width=np.diff(patchesHist.S.values*hectare_coef),
            align='edge')
#        plt.step(x=patchesHist.S.values[:]*hectare_coef,
#                y=patchesHist.N.values[:]*coef,
##                width=np.diff(patchesHist.S.values*hectare_coef),
#                where='post', label=islands_or_expansions)
#        plt.title(str(Tif)+' '+islands_or_expansions)
    plt.grid()
    if hectare:
        plt.xlabel("ha")
    else:
        plt.xlabel("px")
    plt.ylabel("frequency")
#        plt.show()