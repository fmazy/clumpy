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
        return('region_params:'+str(list(self.transitions)))
    
    def add_transitions(self, state, params_transitions):
        """
        Set all transitions for each initial states

        Parameters
        ----------
        state : State
            The final state.
        
        params_transitions : parameters.Transitions
            The transitions parameters.

        Returns
        -------
        self : Region
            The self object.
        """
        self.transitions[state] = params_transitions
        
        return(self)