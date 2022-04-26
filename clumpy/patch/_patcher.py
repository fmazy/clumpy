#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


from copy import deepcopy
from scipy import ndimage, stats

from ..tools._data import np_drop_duplicates_from_column

structures = {
    'queen' : np.ones((3, 3)),
    'rook' : np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])
}

class Patchers(dict):
    
    # def __init__(self, state):
    #     self.state = self.state
    
    def add_patcher(self, patcher):
        self[patcher.state] = patcher
    
    def check(self, objects=None):
        if objects is None:
            objects = []
            
        for state, patcher in self.items():
            if patcher in objects:
                raise(ValueError("Patchers objects must be different."))
            else:
                objects.append(patcher)
            
            if int(state) != int(patcher.final_state):
                raise(ValueError("Patchers keys does not correspond to Patcher().initial_state values."))
            
    
    def fit(self,
            J,
            V,
            shape):
        
        for state, patch in self.items():
            patch.fit(J,
                      V,
                      shape)
    
    def area_mean(self, 
                  final_states):
        
        a = []
        for final_state in final_states:
            if final_state in self.keys():
                a.append(self[final_state].area_mean)
            else:
                a.append(np.nan)
        
        return np.array(a)

class Patcher():
    """
    Patch parameters object. Useful for developers.

    Parameters
    ----------
    neighbors_structure : {'rook', 'queen'}, default='rook'
        The neighbors structure.

    avoid_aggregation : bool, default=True
        If ``True``, the patcher will avoid patch aggregations to respect expected patch areas.

    nb_of_neighbors_to_fill : int, default=3
        The patcher will allocate cells whose the number of allocated neighbors is greater than this integer
        (according to the specified ``neighbors_structure``)

    proceed_even_if_no_probability : bool, default=True
        The patcher will allocate even if the neighbors have no probabilities to transit.

    n_tries_target_sample : int, default=10**3
        Number of tries to draw samples in a biased way in order to approach the mean area.

    equi_neighbors_proba : bool, default=False
        If ``True``, all neighbors have the equiprobability to transit.
    """
    def __init__(self,
                 initial_state,
                 final_state,
                 neighbors_structure = 'rook',
                 avoid_aggregation = True,
                 nb_of_neighbors_to_fill = 3,
                 proceed_even_if_no_probability = True,
                 n_tries_target_sample = 10**3,
                 equi_neighbors_proba = False):
        self.initial_state = initial_state
        self.final_state = final_state
        self.neighbors_structure = neighbors_structure
        self.avoid_aggregation = avoid_aggregation
        self.nb_of_neighbors_to_fill = nb_of_neighbors_to_fill
        self.proceed_even_if_no_probability = proceed_even_if_no_probability
        self.n_tries_target_sample = n_tries_target_sample
        self.equi_neighbors_proba = equi_neighbors_proba

        # for compatibility, set mean area and eccentricities to 1.0 by default.
        self.area_mean = 1.0
        
    
    def __repr__(self):
        return("Patcher("+str(self.initial_state)+"->"+str(self.final_state)+")")
    
    def copy(self):
        return(deepcopy(self))
    
    def sample(self, n=1):
        """
        draws patches.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        areas : ndarray of shape (n_samples,)
            The samples areas.
        eccentricities : ndarray of shape (n_samples,)
            The samples eccentricities.
        """
        areas, eccentricities = self._sample(n)
        if n==1:
            return areas[0], eccentricities[0]
        else:
            return areas, eccentricities
    
    def target_sample(self, n):
        """
        Draw areas and eccentricities according to a targeted total area (biased sample).
    
        Parameters
        ----------
        n : int
            The number of samples.
    
        Returns
        -------
        areas : ndarray of shape (n_samples,)
            The samples areas.
        eccentricities : ndarray of shape (n_samples,)
            The samples eccentricities.
        """
        n_try = 0
            
        best_areas = None
        best_eccentricities = None
        best_relative_error = np.inf
        
        total_area_target = self.area_mean * n
        
        while n_try < self.n_tries_target_sample:
            n_try += 1
            
            areas, eccentricities = self.sample(n)
            
            relative_error = np.abs(total_area_target - areas.sum()) / total_area_target
            
            if relative_error < best_relative_error:
                best_relative_error = relative_error
                best_areas = areas
                best_eccentricities = eccentricities
        
        return(best_areas, best_eccentricities)

    def fit(self,
            J,
            V,
            shape):
        return(self)
    
    def allocate(self,
                 lul,
                 lul_origin,
                 j):
        
        J_allocated = [j]
        x_allocated, y_allocated = lul.unravel_index(J_allocated)
        
        # if the kernel pixel is already transited or if the surface is negative
        if lul.flat[j] != self.initial_state:
            return 0, J_allocated
        
        area, eccentricity = self.sample(n=1)
        
        while len(J_allocated) < area:
            # construction de la fenêtre de voisins
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
            if x_allocated.max() == lul.shape[0] - 1:
                box_shape[0] -= 1
            
            # si bord gauche
            if y_allocated.max() == lul.shape[1] - 1:
                box_shape[1] -= 1
                        
            box_shape = tuple(box_shape)
            
            # on construit la matrice de la tache
            A = np.zeros(box_shape)
            
            A[x_allocated-x_offset, y_allocated-y_offset] = 1
            
            # on détermine les voisins
            B = _convolve(A, structures[self.neighbors_structure])
            
            x_neighbors_box, y_neighbors_box = np.where(B*(1-A) > 0) # C = B*(1-A)
            j_neighbors_box = np.ravel_multi_index([x_neighbors_box, y_neighbors_box], box_shape)
            
            x_neighbors = x_neighbors_box + x_offset
            y_neighbors = y_neighbors_box + y_offset
            
            j_neighbors = lul.ravel_index(x_neighbors, y_neighbors)
            
            vi_neighbors = lul_origin.flat[j_neighbors]
            vf_neighbors = lul.flat[j_neighbors]
            
            # si on veut éviter les aggrégations
            if self.avoid_aggregation and (np.sum((vi_neighbors == self.initial_state) * (vf_neighbors == self.final_state)) > 0):
                # si un voisin a déja subi la transition, il fait échouer la tache
                # print('aggrégation')
                return 0, J_allocated
            
            # on ne garde que les voisins dont l'état initial et l'état final sont à vi
            id_j_neighbors_to_keep = np.arange(j_neighbors.size)[(vi_neighbors == self.initial_state) * (vf_neighbors == self.initial_state)]
            
            # si aucun des voisins n'est convenable, on annule 
            if id_j_neighbors_to_keep.size == 0:
                return 0, J_allocated
            
            j_neighbors_box = j_neighbors_box[id_j_neighbors_to_keep]
            b_neighbors = B.flat[j_neighbors_box]
            j_neighbors = j_neighbors[id_j_neighbors_to_keep]
            x_neighbors = x_neighbors[id_j_neighbors_to_keep]
            y_neighbors = y_neighbors[id_j_neighbors_to_keep]
            
            # si on veut remplir les cuvettes
            if self.nb_of_neighbors_to_fill > 0:
                j_hollows = j_neighbors[b_neighbors >= self.nb_of_neighbors_to_fill]
                if j_hollows.size > 0:
                    J_allocated.append(np.random.choice(j_hollows))
                    continue
            
            # on attribue une probabilité à chaque voisin
            if self.equi_neighbors_proba or True:
                P = np.ones(j_neighbors.size)
            # else:
                # P = map_P_vf__vi_z.flat[j_neighbors]
            
            # si les probas sont nulles, on les met à 1            
            if np.isclose(P.sum(), 0):
                if self.proceed_even_if_no_probability:
                    P.fill(1)
                else:
                    return 0, J_allocated
                                     
            xc = np.sum(x_allocated) / len(J_allocated)
            yc = np.sum(y_allocated) / len(J_allocated)
            
            mu_20 = (np.sum(np.power(x_allocated-xc,2)) + np.power(x_neighbors - xc,2)) / (len(J_allocated)+1)
            mu_02 = (np.sum(np.power(y_allocated-yc,2)) + np.power(y_neighbors - yc,2)) / (len(J_allocated)+1)
            mu_11 = (np.sum((x_allocated-xc)*(y_allocated-yc)) + (x_neighbors - xc) * (y_neighbors - yc)) / (len(J_allocated)+1)
            
            delta = np.power(mu_20-mu_02,2) + 4 * np.power(mu_11,2)
            # l1 = (mu_20+mu_02 + np.sqrt(delta))/2
            # l2 = (mu_20+mu_02 - np.sqrt(delta))/2
            
            # e = 1 - minor axis length / major axis length
            e = 1-np.sqrt((mu_20+mu_02 - np.sqrt(delta))/(mu_20+mu_02 + np.sqrt(delta)))
                        
            eccentricity_coef = stats.norm.pdf(e, 
                                               loc=eccentricity, 
                                               scale=eccentricity * 0.1)
            
            if eccentricity_coef.sum() <= 0:
                eccentricity_coef.fill(1)
            
            P *= eccentricity_coef
            
            if np.isclose(P.sum(), 0):
                if self.proceed_even_if_no_probability:
                    P.fill(1)
                else:
                    return(0, J_allocated)
            
            # sum(P) = 1
            P /= P.sum()
            
            # security to avoid unexpectec nan values
            P = np.nan_to_num(P)
            
            J_allocated.append(np.random.choice(j_neighbors, p=P))
        
        if self.avoid_aggregation:
            # on vérifie que le dernier pixel ajouté n'a pas des voisins qui font échouer la tache
            last_neighbors = lul.get_neighbors_id(j=J_allocated[-1], 
                                                  neighbors_structure=self.neighbors_structure)
            vi_neighbors = lul_origin.flat[last_neighbors]
            vf_neighbors = lul.flat[last_neighbors]
            
            if np.any((vi_neighbors == self.initial_state) & (vf_neighbors == self.final_state)):
                # print('aggrégation finale')
                return(0, J_allocated)
                
        # on peut procéder à l'allocation réelle
        lul.flat[J_allocated] = self.final_state
        return(len(J_allocated), J_allocated)

def _convolve(A,B):
    return(ndimage.convolve(A, B, mode='constant', cval=0))