"""
intro
"""

import numpy as np
import scipy

def _get_neighbors_id(j, shape, neighbors_structure='rook'):
    if neighbors_structure == 'queen':
        j_neighbors = j + np.array([- shape[1],     # 0, top
                                    - shape[1] + 1, # 1, top-right
                                      1,            # 2, right
                                      shape[1] + 1, # 3, bottom-right
                                      shape[1],     # 4, bottom
                                      shape[1] - 1, # 5, bottom-left
                                    - 1,            # 6, left
                                    - shape[1] - 1])# 7, top-left
        
        # remove if side pixel
        if (j + 1) % shape[1] == 0: # right side
            j_neighbors = np.delete(j_neighbors, [1,2,3])
        if j % shape[1] == 0: # left side
            j_neighbors = np.delete(j_neighbors, [5,6,7])
        if j >= shape[0]*shape[1] - shape[1]: # bottom side
            j_neighbors = np.delete(j_neighbors, [3,4,5])
        if j < shape[1]: # top side
            j_neighbors = np.delete(j_neighbors, [0,1,7])
            
    elif neighbors_structure == 'rook':
        j_neighbors = j + np.array([- shape[1],     # 0, top
                                      1,            # 1, right
                                      shape[1],     # 2, bottom
                                    - 1])           # 3, left
        
        # remove if side pixel
        if (j + 1) % shape[1] == 0: # right side
            j_neighbors = np.delete(j_neighbors, [1])
        if j % shape[1] == 0: # left side
            j_neighbors = np.delete(j_neighbors, [3])
        if j >= shape[0]*shape[1] - shape[1]: # bottom side
            j_neighbors = np.delete(j_neighbors, [2])
        if j < shape[1]: # top side
            j_neighbors = np.delete(j_neighbors, [0])
    else:
        print('ERROR, unexpected neighbors_structure')
    
    return(j_neighbors)

def _convolve(A,B):
    return(scipy.ndimage.convolve(A, B, mode='constant', cval=0))

def _weighted_neighbors(map_i_data,
                           map_f_data,
                           map_P_vf__vi_z,
                           j_kernel,
                           vi,
                           vf,
                           patch_S,
                           eccentricity_mean=None,
                           eccentricity_std=None,
                           neighbors_structure = 'rook',
                           avoid_aggregation = True,
                           nb_of_neighbors_to_fill = 3,
                           proceed_even_if_no_probability=True):
    # if the kernel pixel is already transited or if the surface is negative
    if (map_f_data.flat[j_kernel] != vi) or (patch_S <= 0):
        return(0)
    
    if neighbors_structure not in ['rook', 'queen']:
        print('ERROR: unexpected neighbors_structure in weighted_neighbors')
        return('ERROR')
    
    queen_star = np.ones((3,3))
    queen_star[1,1] = 0
        
    rook_star = np.array([[0,1,0],
                          [1,0,1],
                          [0,1,0]])
    
    j_allocated = [j_kernel]
        
    while len(j_allocated) < patch_S:
        x_allocated, y_allocated = np.unravel_index(j_allocated, map_f_data.shape)
        
        # de manière générale,
        box_shape = [x_allocated.max()-x_allocated.min()+3,
                      y_allocated.max()-y_allocated.min()+3]
        x_offset = x_allocated.min() - 1
        y_offset = y_allocated.min() - 1
    
        # si bord haut
        if x_allocated.min() == 0:
            x_offset += 1
            box_shape[0] -= 1
        
        # si bord gauche
        if y_allocated.min() == 0:
            y_offset += 1
            box_shape[1] -= 1
            
        # si bord droit
        if x_allocated.max() == map_f_data.shape[0] - 1:
            box_shape[0] -= 1
        
        # si bord gauche
        if y_allocated.max() == map_f_data.shape[1] - 1:
            box_shape[1] -= 1
                    
        box_shape = tuple(box_shape)
        
        # on construit la matrice de la tache
        A = np.zeros(box_shape)
        
        A[x_allocated-x_offset, y_allocated-y_offset] = 1
        
        # on détermine les voisins
        if neighbors_structure == 'rook':
            B = _convolve(A, rook_star)
        else:
            B = _convolve(A, queen_star)
        
        x_neighbors_box, y_neighbors_box = np.where(B*(1-A) > 0) # C = B*(1-A)
        j_neighbors_box = np.ravel_multi_index([x_neighbors_box, y_neighbors_box], box_shape)
        
        x_neighbors = x_neighbors_box + x_offset
        y_neighbors = y_neighbors_box + y_offset
        
        j_neighbors = np.ravel_multi_index([x_neighbors, y_neighbors], map_f_data.shape)
        
        vi_neighbors = map_i_data.flat[j_neighbors]
        vf_neighbors = map_f_data.flat[j_neighbors]
        
        # si on veut éviter les aggrégations
        if (avoid_aggregation) and (np.sum((vi_neighbors == vi) * (vf_neighbors == vf)) > 0):
            # si un voisin a déja subi la transition, il fait échouer la tache
            return(0)
        
        # on ne garde que les voisins dont l'état initial et l'état final sont à vi
        id_j_neighbors_to_keep = np.arange(j_neighbors.size)[(vi_neighbors == vi) * (vf_neighbors == vi)]
        
        # si aucun des voisins n'est convenable, on annule 
        if id_j_neighbors_to_keep.size == 0:
            return(0)
        
        j_neighbors_box = j_neighbors_box[id_j_neighbors_to_keep]
        b_neighbors = B.flat[j_neighbors_box]
        j_neighbors = j_neighbors[id_j_neighbors_to_keep]
        x_neighbors = x_neighbors[id_j_neighbors_to_keep]
        y_neighbors = y_neighbors[id_j_neighbors_to_keep]
        
        # si on veut remplir les cuvettes
        if nb_of_neighbors_to_fill > 0:
            j_hollows = j_neighbors[b_neighbors >= nb_of_neighbors_to_fill]
            if j_hollows.size > 0:
                j_allocated.append(np.random.choice(j_hollows, size=1))
                continue
        
        # on attribue une probabilité à chaque voisin
        P = map_P_vf__vi_z.flat[j_neighbors]
        
        # si les probas sont nulles, on les met à 1            
        if P.sum() <= 0:
            if proceed_even_if_no_probability:
                P.fill(1)
            else:
                return(0)
                
        if type(eccentricity_mean) != type(None):                        
            xc = np.sum(x_allocated) / len(j_allocated)
            yc = np.sum(y_allocated) / len(j_allocated)
            
            mu_20 = (np.sum(np.power(x_allocated-xc,2)) + np.power(x_neighbors - xc,2)) / (len(j_allocated)+1)
            mu_02 = (np.sum(np.power(y_allocated-yc,2)) + np.power(y_neighbors - yc,2)) / (len(j_allocated)+1)
            mu_11 = (np.sum((x_allocated-xc)*(y_allocated-yc)) + (x_neighbors - xc) * (y_neighbors - yc)) / (len(j_allocated)+1)
            
            delta = np.power(mu_20-mu_02,2) +4 * np.power(mu_11,2)
            # l1 = (mu_20+mu_02 + np.sqrt(delta))/2
            # l2 = (mu_20+mu_02 - np.sqrt(delta))/2
            
            # e = 1 - minor axis length / major axis length
            e = 1-np.sqrt((mu_20+mu_02 - np.sqrt(delta))/(mu_20+mu_02 + np.sqrt(delta)))
            
            eccentricity_coef = scipy.stats.norm.pdf(e, loc=eccentricity_mean, scale=eccentricity_std)
            
            if eccentricity_coef.sum() <= 0:
                eccentricity_coef.fill(1)
            
            P *= eccentricity_coef
            
        if P.sum() <= 0:
            if proceed_even_if_no_probability:
                P.fill(1)
            else:
                return(0)
        
        # sum(P) = 1
        P /= P.sum()
        
        j_allocated.append(np.random.choice(j_neighbors, p=P))
    
    if avoid_aggregation:
        # on vérifie que le dernier pixel ajouté n'a pas des voisins qui font échouer la tache
        last_neighbors = _get_neighbors_id(j_allocated[-1], map_f_data.shape, neighbors_structure='rook')
        vi_neighbors = map_i_data.flat[last_neighbors]
        vf_neighbors = map_f_data.flat[last_neighbors]
        if np.sum((vi_neighbors == vi) * (vf_neighbors == vf)) > 0:
            return(0)
        
    # on peut procéder à l'allocation réelle
    map_f_data.flat[j_allocated] = vf
    return(len(j_allocated))
 