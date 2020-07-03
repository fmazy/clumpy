"""
intro
"""

from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy import ndimage # used for distance computing and patch measurements
from matplotlib import pyplot as plt
#from PIL import Image
#from scipy import signal, sparse
from skimage import measure # for patch perimeters

from ..definition import _transition
from .. import tools as dmtools
from .. import definition


def analyse(map_i, map_f, list_vi_vf, neighbors_structure = 'queen'):
    map_shape = np.shape(map_i.data)
    
    J = definition.data.create_J(map_i, map_f)
    
    if neighbors_structure == 'queen':
        structure = np.ones((3,3))
    elif neighbors_structure == 'rook':
        structure = np.array([[0,1,0],
                              [1,1,1],
                              [0,1,0]])
    else:
        print('ERROR : unexpected neighbors_structure value')
        return(False)
    
    list_vi = []
    list_vf = []
    for vi_vf in list_vi_vf:
        list_vi.append(vi_vf[0])
        list_vf.append(vi_vf[1])
    
    J_vf = J.loc[J.v.f.isin(list_vf)].copy()
    
    J['vi_vf'] = list(zip(J.v.i, J.v.f))
    J_vi_vf = J.loc[J.vi_vf.isin(list_vi_vf)].copy()    
      
    J_vi_vf.drop('vi_vf', axis=1, level=0, inplace=True)
    
    J_vf[('id_patch','vf')] = -1
    patches_vf = pd.DataFrame()
    for vf in list_vf:
        # on récupère la carte vf
        j_vf = J_vf.loc[(J_vf.v.f == vf)].index.values
        
        M_vf = np.zeros(map_shape)
        M_vf.flat[j_vf] = 1
        
        # on donne à chaque ilot un id
        lw_vf, _ = ndimage.measurements.label(M_vf, structure=structure)
        J_vf.loc[j_vf, [('id_patch','vf')]] = lw_vf.flat[j_vf]
        
        # # paramètres des taches
        p= pd.DataFrame(measure.regionprops_table(lw_vf, properties=['label','area']))
        p.columns = pd.MultiIndex.from_tuples([('id_patch','vf'), ('S','vf')])
        p[('v','f')] = vf
        patches_vf = pd.concat([patches_vf, p], ignore_index=True)

    J_vf.rename({'index':'j'}, level=0, axis=1, inplace=True)
    J_vf.reset_index(drop=False, inplace=True)
    J_vf.rename({'index':'j'}, level=0, axis=1, inplace=True)
    J_vf = J_vf.merge(right=patches_vf.astype(float), how='left')
    J_vf.drop('id_patch', axis=1, level=0, inplace=True)
        
    J_vi_vf[('id_patch','vi_vf')] = -1
    patches_vi_vf = pd.DataFrame()
    
    for vi_vf in list_vi_vf:
        vi = vi_vf[0]
        vf = vi_vf[1]
        
        # on fait comme pour vf mais pour vi -> vf 
        j_vi_vf = J_vi_vf.loc[(J_vi_vf.v.i == vi) & (J_vi_vf.v.f == vf)].index.values
        
        M_vi_vf = np.zeros(map_shape)
        M_vi_vf.flat[j_vi_vf] = 1
        
        # on donne à chaque ilot un id
        lw_vi_vf, _ = ndimage.measurements.label(M_vi_vf, structure=structure)
        J_vi_vf.loc[j_vi_vf, [('id_patch','vi_vf')]] = lw_vi_vf.flat[j_vi_vf]
        
        # paramètres des taches
        p= pd.DataFrame(measure.regionprops_table(lw_vi_vf, properties=['label',
                                                                        'area',
                                                                        'centroid',
                                                                        'inertia_tensor_eigvals']))
        p.columns = pd.MultiIndex.from_tuples([('id_patch','vi_vf'),
                                               ('parameters','area'),
                                               ('parameters','centroid_x'),
                                               ('parameters','centroid_y'),
                                               ('parameters','l1'),
                                               ('parameters','l2')])
        p[('v','i')] = vi
        p[('v','f')] = vf
        patches_vi_vf = pd.concat([patches_vi_vf, p], ignore_index=True)
    
    J_vi_vf.reset_index(drop=False, inplace=True)
    J_vi_vf.rename({'index':'j'}, level=0, axis=1, inplace=True)
    J_vi_vf = J_vi_vf.merge(right=patches_vi_vf.astype(float), how='left')
    J_vi_vf.set_index('j', inplace=True)
        
    patches = J_vi_vf.drop_duplicates().copy()
    
    patches.reset_index(drop=False, inplace=True)
    patches.rename({'index':'j'}, level=0, axis=1, inplace=True)
        
    patches = patches.merge(right=J_vf, how='left')    
    
    patches[('parameters', 'isl_exp')] = 'exp'
    patches.loc[patches.parameters.area == patches.S.vf, ('parameters', 'isl_exp')] = 'isl'
    patches.drop(columns=('S','vf'), inplace=True)
    
    patches[('parameters','eccentricity')] = 1-np.sqrt(patches.parameters.l2 / patches.parameters.l1)
    
    patches.drop(columns=[('parameters', 'l1'), ('parameters','l2')], inplace=True)
    
    return(patches)


def analyse_surfaces(map_i, map_f, T):
        
    # mériterait d'être nettoyé mais ça fonctionne bien !
    
    map_shape = np.shape(map_i.data)
    
    surfaces = pd.DataFrame(columns=['vi', 'vf', 'S', 'isl_exp'])
    

    
    # all vf possible
    list_vf = T.get_all_possible_vf()
    
    J = pd.DataFrame(map_i.data.flat, columns=['vi'])
    J = J.assign(vf = map_f.data.flat)
    
    # restrict to pixels with vi in Ti
    J = J.loc[J.vf.isin(list_vf)]
    
    for Ti in T.Ti.values():
        for Tif in Ti.Tif.values():
            
            # patch identification for vi->vf
            J_vi_vf = J.loc[(J.vi == Ti.vi) &
                            (J.vf == Tif.vf)]
            
            M_vi_vf = np.zeros(map_shape)
            M_vi_vf.flat[J_vi_vf.index.values] = 1
            
            lw_vi_vf, _ = ndimage.measurements.label(M_vi_vf) # création des îlots
            J_vi_vf = J_vi_vf.assign(id_patch = lw_vi_vf.flat[J_vi_vf.index.values])
            
            # patch identification for * -> vf
            J_vf = J.loc[(J.vf == Tif.vf)]
            J_vf = J_vf.drop('vi', axis=1)
            
            M_vf = np.zeros(map_shape)
            M_vf.flat[J_vf.index.values] = 1
            
            lw_vf, _ = ndimage.measurements.label(M_vf) # création des îlots
            J_vf = J_vf.assign(id_patch = lw_vf.flat[J_vf.index.values])
            
            # in vi-> vf compute S and drop duplicates in J_vi_vf
            S = J_vi_vf.groupby(['id_patch']).size().reset_index(name='S')
            
            J_vi_vf = J_vi_vf.reset_index().merge(right= S, how='left', on='id_patch').set_index('index')
            J_vi_vf.drop_duplicates(inplace=True)
            
            # get id_patch from J_vf
            J_vi_vf = J_vi_vf.assign(id_patch_J_vf = J_vf.loc[J_vi_vf.index, 'id_patch'])
            
            # compute S and drop duplicates in J_vf
            S = J_vf.groupby(['id_patch']).size().reset_index(name='S_vf')
            
            J_vf = J_vf.reset_index().merge(right= S, how='left', on='id_patch').set_index('index')
            J_vf.drop_duplicates(inplace=True)
            
            # merge with vi vf
            J_vi_vf = J_vi_vf.merge(right=J_vf[['id_patch', 'S_vf']],
                                    how='left',
                                    left_on='id_patch_J_vf',
                                    right_on='id_patch')
            
            # comparison
            J_vi_vf = J_vi_vf.assign(S_diff=J_vi_vf.S-J_vi_vf.S_vf)
            
            # island or expansion ?
            J_vi_vf = J_vi_vf.assign(isl_exp = 'exp')
            J_vi_vf.loc[J_vi_vf.S_diff == 0, 'isl_exp'] = 'isl'
            
            surfaces = pd.concat([surfaces, J_vi_vf[['vi', 'vf', 'S', 'isl_exp']]], ignore_index = True)
    
    cols = [('v', 'i'), ('v', 'f'), ('patch','S'), ('patch', 'isl_exp')]
    cols = pd.MultiIndex.from_tuples(cols)
    surfaces.columns = cols
    return(surfaces)

def surfaces_histogram(surfaces, T):
    surfaces_histogram = pd.DataFrame(columns=['vi', 'vf', 'isl_exp', 'S', 'N'])
    for Ti in T.Ti.values():
        for Tif in Ti.Tif.values():
            for isl_exp in ['isl', 'exp']:
                N, S = np.histogram(a=surfaces.loc[(surfaces.vi == Ti.vi) &
                                                   (surfaces.vf == Tif.vf) &
                                                   (surfaces.isl_exp == isl_exp)].S.values,
                                                    bins='auto',
                                                    density=True)
                
                N = np.append(N, 0) # faut que les listes aient la même longueur. ça conclut les bins.
                
                sub_surfaces_histogram = pd.DataFrame(columns=['vi', 'vf', 'isl_exp', 'S', 'N'])
                sub_surfaces_histogram.S = S
                sub_surfaces_histogram.N = N
                sub_surfaces_histogram.vi = Tif.Ti.vi
                sub_surfaces_histogram.vf = Tif.vf
                sub_surfaces_histogram.isl_exp = isl_exp
                
                surfaces_histogram = pd.concat([surfaces_histogram,
                                                sub_surfaces_histogram], ignore_index=True)
                
    return(surfaces_histogram)

def plot_surfaces_histogram(surfaces_histogram, vi, vf, isl_exp, color=None, linestyle=None, linewidth=None, label=None):
    part = surfaces_histogram.loc[(surfaces_histogram.vi == vi) &
                               (surfaces_histogram.vf == vf) &
                               (surfaces_histogram.isl_exp == isl_exp)]
    
    S = part.S.values
    N = part.N.values
    
    plt.step(x=S,
            y=N,
            where='post',
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label)

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