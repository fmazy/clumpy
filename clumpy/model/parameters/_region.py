#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Region():
    """
    Region parameters.

    Parameters
    ----------
    transition_matrix : TransitionMatrix
        The transition matrix for this region.
    """
    def __init__(self, transition_matrix):
        self.transition_matrix = transition_matrix
        self.transitions = {}
        
    def __repr__(self):
        return('region:'+str(list(self.transitions)))
    
    def add_transitions(self, state, transitions):
        """
        Set all transitions for each initial states

        Parameters
        ----------
        transitions : dict of parameters.Transitions with States as keys
            Dict of transitions with states as keys.

        Returns
        -------
        None.

        """
        self.transitions[state] = transitions