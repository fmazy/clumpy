#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from ._state import Palette

class TransitionMatrix():
    """
    Define a transition matrix.

    Parameters
    ----------
    M : array-like of shape (n_initial_states, n_final_states)
        The transition matrix values.
    palette_u : Palette
        List of initial states.
    palette_v : Palette
        List of final states.
    """
    def __init__(self, M, palette_u=None, palette_v=None):
        
        M = np.nan_to_num(M)
        
        # check
        if not np.all(np.isclose(np.sum(M, axis=1), np.ones(M.shape[0]))):
            print("Warning : The transition matrix is uncorrect. The rows should sum to one")
        
        self.M = M
        self.palette_u = palette_u
        self.palette_v = palette_v
    
    def __repr__(self):
        return('TM:'+repr(self.palette_u)+'->'+repr(self.palette_v))
    
    def get(self, info_u, info_v):
        """
        Get a probability.

        Parameters
        ----------
        info_u : State or int or str
            The initial state information which can be the object, the state's value or the state's label.
            If two states share the same label, only the first one to occur is returned.
        info_v : State or int or str
            The final state information which can be the object, the state's value or the state's label.
            If two states share the same label, only the first one to occur is returned.

        Returns
        -------
        p : float
            The requested probability.
        """
        id_u = self.palette_u.get_id(info_u)
        id_v = self.palette_v.get_id(info_v)
        
        return(self.M[id_u, id_v])
    
    def _get_by_states(self, state_u, state_v):
        """
        Get a probability by states.

        Parameters
        ----------
        state_u : State
            Initial state.
        state_v : State
            Final state.

        Returns
        -------
        p : float
            The requested probability.^
        """
        id_u = self.palette_u._get_id(state_u)
        id_v = self.palette_v._get_id(state_v)
        
        return(self.M[id_u, id_v])
    
    def _get_by_states_values(self, u, v):
        """
        Get probability by states values.

        Parameters
        ----------
        u : int
            Initial state value.
        v : int
            Final state value.

        Returns
        -------
        p : float
            The requested probability.
        """
        id_u = self.palette_u.get_id_by_value(u)
        id_v = self.palette_v.get_id_by_value(v)
        
        return(self.M[id_u, id_v])
    
    def copy(self):
        """
        Make a copy.

        Returns
        -------
        tm : TransitionMatrix
            The copy.

        """
        return(TransitionMatrix(self.M.copy(), self.palette_u, self.palette_v))
    
    def set_value(self, a, info_u, info_v):
        """
        Set a transition probability value.

        Parameters
        ----------
        a : float
            The transition probability value.
        info_u : State or int or str
            The initial state information which can be the object, the state's value or the state's label.
            If two states share the same label, only the first one to occur is returned.
        v : State or int or str
            The final state information which can be the object, the state's value or the state's label.
            If two states share the same label, only the first one to occur is returned.
        """
        id_u = self.palette_u.get_id(info_u)
        id_v = self.palette_v.get_id(info_v)
        
        self.M[id_u, id_v] = a
    
    def select_land(self, state):
        """
        Select a new transition probabilities object for a given initial state.

        Parameters
        ----------
        state : State or int or str
            The initial state information which can be the object, the state's value or the state's label.
            If two states share the same label, only the first one to occur is returned.

        Returns
        -------
        tm : TransitionMatrix
            The selected transition matrix object.

        """
        id_u = self.palette_u.get_id(state)
        
        M = self.M[[id_u],:].copy()
                
        new_palette_u = Palette([self.palette_u.states[id_u]])
        
        return(TransitionMatrix(M, new_palette_u, self.palette_v))
    
    def get_P_v(self, info_u):
        """
        Get all transition probabilities from a given initial state.

        Parameters
        ----------
        info_u : State or int or str
            The initial state information which can be the object, the state's value or the state's label.
            If two states share the same label, only the first one to occur is returned.

        Returns
        -------
        P_v : ndarray of shape (n_final_states,)
            The requested transition probabilities.
        palette_v : Palette
            The corresponding final states palette.
        """
        id_u = self.palette_u.get_id(info_u)
        
        p = self.M[id_u, :].copy()
        
        return(p, self.palette_v)
            
    def multisteps(self, n):
        """
        Computes multisteps transition matrix.
        
        Parameters
        ----------
        n : int
            Number of steps.
        
        Returns
        -------
        tm : TransitionMatrix
            The computed multistep transition matrix.
        """
        tp = self._full_matrix()
        
        eigen_values, P = np.linalg.eig(tp)
        
        eigen_values = np.power(eigen_values, 1/n)
        
        full_M = np.dot(np.dot(P, np.diag(eigen_values)), np.linalg.inv(P))
        
        list_u = self.palette_u.get_list_of_values()
        list_v = self.palette_v.get_list_of_values()
        
        compact_M, _, _ = _compact_transition_matrix(full_M = full_M,
                                               list_u = list_u,
                                               list_v = list_v)
        
        return(TransitionMatrix(compact_M, self.palette_u, self.palette_v))
    
    def patches(self, patches):
        """
        Computes a new transition matrix according to patches parameters.

        Parameters
        ----------
        patches : TYPE
            DESCRIPTION.

        Returns
        -------
        tm : TransitionMatrix
            The transition matrix computed according to patches.

        """
        M_p = np.zeros_like(self.M)
        
        for id_u, u in enumerate(self.list_u):
            M_p[id_u, self.list_v.index(u)] = 1.0
            for id_v, v in enumerate(self.list_v):
                if u != v:
                    if patches[u][v]['area'].size > 0 and self.M[id_u, id_v] > 0:
                        M_p[id_u, id_v] = self.M[id_u, id_v] / patches[u][v]['area'].mean()
                        M_p[id_u, self.list_v.index(u)] -= M_p[id_u, id_v]
        
        return(TransitionMatrix(M_p, self.list_u, self.list_v))
    
    def _full_matrix(self):
        """
        Get the full matrix

        Returns
        -------
        full_M : ndarray of shape (max_palette_v_values, max_palette_v_values)
            The full transition probabilities matrix.
        """
        list_u = self.palette_u.get_list_of_values()
        list_v = self.palette_v.get_list_of_values()
        
        max_v = np.max(self.palette_v.get_list_of_values())
        
        full_M = np.diag(np.ones(max_v+1))
    
        list_v_full_matrix = list(np.arange(max_v+1))
    
        for id_u, u in enumerate(list_u):
            for id_v, v in enumerate(list_v):
                full_M[list_v_full_matrix.index(u),
                       list_v_full_matrix.index(v)] = self.M[id_u, id_v]
        
        full_M = full_M.astype(float)
        
        full_M = np.nan_to_num(full_M)
        
        return(full_M)

def compute_transition_matrix(V_state, palette_u, palette_v):
    """
    Compute the transition matrix from final states vectors.

    Parameters
    ----------

    """
    list_u = palette_u.get_list_of_values()
    list_v = palette_v.get_list_of_values()
    
    M = np.zeros((len(list_u), len(list_v)))
    
    for id_u, u in enumerate(list_u):
        for id_v, v in enumerate(list_v):
            M[id_u, id_v] = np.mean(V_u[u] == v)
    
    return(TransitionMatrix(M, palette_u, palette_v))

def load_transition_matrix(path, palette):
    """
    Load transition matrixes according to a given palette.

    Parameters
    ----------
    path : str
        Transition matrix file path.

    Returns
    -------
    tm : TransitionMatrix
        The loaded transition matrix.

    """
    data = np.genfromtxt(path, delimiter=',')
    
    list_u = list(data[1:,0].astype(int))
    list_v = list(data[0,1:].astype(int))

    M = data[1:,1:]
    
    return(TransitionMatrix(M, list_u, list_v))

def _compact_transition_matrix(full_M, list_u=None, list_v=None):
    """
    Extract compact transition matrix from full_M
    """
    if list_u is None or list_v is None:
        list_u = []
        list_v = []
        
        for u in range(full_M.shape[0]):
            v_not_null = np.arange(full_M.shape[1])[full_M[u, :] > 0]
            
            # if transitions (u -> u is not a transition)
            if len(v_not_null) > 1:
                if u not in list_u:
                        list_u.append(u)
                for v in v_not_null:
                    if v not in list_v:
                        list_v.append(v)
        
        list_u.sort()
        list_v.sort()
        
    list_v_full_matrix = list(np.arange(full_M.shape[0]))
        
    compact_M = np.zeros((len(list_u), len(list_v)))
    
    for id_u, u in enumerate(list_u):
        for id_v, v in enumerate(list_v):
            compact_M[id_u, id_v] = full_M[list_v_full_matrix.index(u),
                                           list_v_full_matrix.index(v)]
    
    return(compact_M, list_u, list_v)

