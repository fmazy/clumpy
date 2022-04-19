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

    def __init__(self, M, palette_u, palette_v):

        if len(palette_u) != M.shape[0] or len(palette_v) != M.shape[1]:
            raise (ValueError('palette_u (len=' + str(len(palette_u)) + ') and palette_v (len=' + str(
                len(palette_v)) + ') should describe exactly M.shape=' + str(M.shape) + ' !'))

        M = np.nan_to_num(M)

        # check
        if not np.all(np.isclose(np.sum(M, axis=1), np.ones(M.shape[0]))):
            print("Warning : The transition matrix is uncorrect. The rows should sum to one")

        self.M = M
        self.palette_u = palette_u
        self.palette_v = palette_v

    def __repr__(self):
        return ('TM:' + repr(self.palette_u) + '->' + repr(self.palette_v))

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

        return (self.M[id_u, id_v])

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

        return (self.M[id_u, id_v])

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

        return (self.M[id_u, id_v])

    def copy(self):
        """
        Make a copy.

        Returns
        -------
        tm : TransitionMatrix
            The copy.

        """
        return (TransitionMatrix(self.M.copy(), self.palette_u, self.palette_v))

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

    def extract(self, infos):
        """
        Select a new transition probabilities object for a given initial state.

        Parameters
        ----------
        infos : int or state or list(int or State)
            Initial state information which can be the object, the state's value or the state's label or a list of them. It will constitutes the initial palette.

        Returns
        -------
        tm : TransitionMatrix
            The extracted transition matrix object.
        """
        if not isinstance(infos, list):
            infos = [infos]

        states_id = [self.palette_u.get_id(info) for info in infos]
        states = [self.palette_u.states[i] for i in states_id]

        M = self.M[states_id, :].copy()

        return (TransitionMatrix(M, Palette(states), self.palette_v))

    def merge(self, tm, inplace=False):
        """
        Merge the transition matrix to another.

        Parameters
        ----------
        tm : TransitionMatrix
            The transition matrix to join

        inplace : bool, default=True
            Inplace operation

        Returns
        -------
        tm : TransitionMatrix
            The merged transition matrix. If ``inplace=True``, it is the self object. Else, it is a new object.
        """
        # if one is empty:
        if len(self.palette_u) == 0:
            palette_u = tm.palette_u
            palette_v = tm.palette_v
            M = tm.M
            if inplace:
                self.palette_u = palette_u
                self.palette_v = palette_v
                self.M = M
                return (self)
            else:
                return (TransitionMatrix(M=M,
                                         palette_u=palette_u,
                                         palette_v=palette_v))

        if len(tm.palette_v) == 0:
            if inplace:
                return (self)
            else:
                return (TransitionMatrix(M=self.M,
                                         palette_u=self.palette_u,
                                         palette_v=self.palette_v))

        # no empty TM. merge process :
        full_self_M, list_v_full_self = self._full_matrix()
        full_tm_M, list_v_full_tm = tm._full_matrix()

        n_v = np.max([len(list_v_full_self), len(list_v_full_tm)])

        full_M = np.zeros((n_v, n_v))
        for A in [full_self_M, full_tm_M]:
            full_M[:A.shape[0], :A.shape[1]] += A

        # fill diagonal to have sum(axis=1) = 1
        np.fill_diagonal(full_M, 0)
        np.fill_diagonal(full_M, 1 - full_M.sum(axis=1))

        palette_u = self.palette_u.merge(tm.palette_u, inplace=False)
        palette_v = self.palette_v.merge(tm.palette_v, inplace=False)

        merged_tm = _full_M_to_transition_matrix(full_M, palette_u, palette_v)

        if inplace:
            self.M = merged_tm.M
            self.palette_u = palette_u
            self.palette_v = palette_v

            return (self)
        else:
            return (merged_tm)

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

        return (p, self.palette_v)
    
    def get_final_palette(self, info_u):
        p = self.get_P_v(info_u)[0]
        id_possible_v = np.arange(p.size)[p > 0]
        final_palette = Palette(states=[self.palette_v.states[id_v] for id_v in id_possible_v])
        
        return(final_palette)

    def multisteps(self, n, inplace=False):
        """
        Computes multisteps transition matrix.
        
        Parameters
        ----------
        n : int
            Number of steps.

        inplace : bool, default=True
            Inplace operation
        
        Returns
        -------
        tm : TransitionMatrix
            The multistep transition matrix. If ``inplace=True``, it is the self object. Else, it is a new object.
        """
        tp, _ = self._full_matrix()

        eigen_values, P = np.linalg.eig(tp)

        eigen_values = np.power(eigen_values, 1 / n)

        full_M = np.dot(np.dot(P, np.diag(eigen_values)), np.linalg.inv(P))

        multisteps_tm = _full_M_to_transition_matrix(full_M=full_M,
                                                     palette_u=self.palette_u,
                                                     palette_v=self.palette_v)

        if inplace:
            self.M = multisteps_tm.M
            return (self)
        else:
            return (multisteps_tm)

    def patches(self, patches, inplace=False):
        """
        divide transition matrix by patches mean area. Useful for allocators.
        Only available for land transition matrix.

        Parameters
        ----------
        patches : dict(State:Patch)
            Dict of patches with states as keys.

        inplace : bool, default=True
            Inplace operation

        Returns
        -------
        tm : TransitionMatrix
            The computed transition matrix. If ``inplace=True``, it is the self object. Else, it is a new object.
        """
        # P_v is divided by patch area mean
        # P_v_patches is then largely smaller.
        # one keep P_v to update it after the allocation try.

        self._check_land_transition_matrix()

        # get the unique palette_u state
        state_u = self.palette_u.states[0]
        # get the index of state_u in palette_v
        id_state = self.palette_v.get_id(state_u)

        M_patches = self.M.copy()
        M_patches[0, id_state] = 1
        for id_state_v, state_v in enumerate(self.palette_v):
            if state_v != state_u:
                if state_v in patches.keys():
                    M_patches[0, id_state_v] /= patches[state_v].area_mean
                M_patches[0, id_state] -= M_patches[0, id_state_v]

        if inplace:
            self.M = M_patches
            return (self)
        else:
            return (TransitionMatrix(M=M_patches,
                                     palette_u=self.palette_u,
                                     palette_v=self.palette_v))

    def _full_matrix(self):
        """
        Get the full matrix

        Returns
        -------
        full_M : ndarray of shape (max_palette_v_values, max_palette_v_values)
            The full transition probabilities matrix.

        list_v_full : list(int)
            list of final state values. Some values may not refers to any existing state.
        """
        list_u = self.palette_u.get_list_of_values()
        list_v = self.palette_v.get_list_of_values()

        max_v = np.max(self.palette_v.get_list_of_values())

        full_M = np.diag(np.ones(max_v + 1))

        list_v_full_matrix = list(np.arange(max_v + 1))

        for id_u, u in enumerate(list_u):
            for id_v, v in enumerate(list_v):
                full_M[list_v_full_matrix.index(u),
                       list_v_full_matrix.index(v)] = self.M[id_u, id_v]

        full_M = full_M.astype(float)

        full_M = np.nan_to_num(full_M)

        return (full_M, list_v_full_matrix)

    def _check_land_transition_matrix(self):
        if len(self.palette_u) != 1 or self.M.shape[0] != 1:
            raise (ValueError(
                "Unexpected transition matrix. Expected a land transition matrix with only one initial state."))
        return(True)


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

    palette_u = palette.extract(infos=list(data[1:, 0].astype(int)))
    palette_v = palette.extract(infos=list(data[0, 1:].astype(int)))

    M = data[1:, 1:]

    return (TransitionMatrix(M=M,
                             palette_u=palette_u,
                             palette_v=palette_v))


def _full_M_to_transition_matrix(full_M, palette_u, palette_v):
    """
    Extract compact transition matrix from full_M
    """
    M = np.zeros((len(palette_u), len(palette_v)))

    for id_u, state_u in enumerate(palette_u):
        u = state_u.value
        for id_v, state_v in enumerate(palette_v):
            v = state_v.value
            M[id_u, id_v] = full_M[u, v]

    return (TransitionMatrix(M=M, palette_u=palette_u, palette_v=palette_v))

    list_u = palette_u.get_list_of_values()
    list_v = palette_v.get_list_of_values()

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

    return (compact_M)
