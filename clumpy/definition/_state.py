#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 22:39:48 2021

@author: frem
"""

from xml.dom import minidom
import numpy as np

class State():
    """
    Define a land use state.        

    Parameters
    ----------
    label : str
        State label. It should be unique.
    value : int
        State value. It must be unique.
    color : str
        State color. It should be unique.
    """
    def __init__(self, label, value, color):
        self.label = label
        self.value = value
        self.color = color
    
    def __repr__(self):
        return(self.label)
        
class Palette():
    """
    Define a palette, i.e. a set of states.

    Parameters
    ----------
    states : list of States, default=None
        A list of States object. Ignored if None.
    """
    def __init__(self, states=None):
        if states is None:
            self.states = []
        else:
            self.states = states
    
    def __repr__(self):
        return(str(self.states))
        
    def add_state(self, state):
        """
        Append a state to the palette's states.

        Parameters
        ----------
        state : State
            The state to append.

        Returns
        -------
        self : Palette
            The self object.

        """
        for s in self.states:
            if s.value == state.value:
                raise(ValueError('Unexpected state : its value is already set'))
    
        self.states.append(state)
        
        return(self)
    
    def get_state_by_value(self, value):
        """
        Get a state by its value.

        Parameters
        ----------
        value : int
            The researched value.

        Returns
        -------
        state : State
            The requested state.
        """
        values = [state.value for state in self.states]
        
        return(self.states[values.index(value)])
    
    def get_state_by_label(self, label):
        """
        Get a state by its label.
        If two states have the same label, only the first one is returned.

        Parameters
        ----------
        label : int
            The researched label.

        Returns
        -------
        state : State
            The requested state.
        """
        labels = [state.label for state in self.states]
        
        return(self.states[labels.index(label)])
    def remove_state(self, state):
        """
        Remove a state from the palette.

        Parameters
        ----------
        state : State
            The state to remove from the palette.

        Returns
        -------
        self : Palette
            The self object.
        """
        self.states.remove(state)
        
        return(self)
    
    def remove_state_by_value(self, value):
        """
        Remove a state by its value

        Parameters
        ----------
        value : int
            The value of the state to remove from the palette.

        Returns
        -------
        self : Palette
            The self object.
        """
        for s in self.states:
            if s.value == value:
                break
        
        self.states.remove(s)
        
        return(self)
    
    def import_style(self, path):
        """
        Import legend through qml file provided by QGis. Alpha is not supported.

        Parameters
        ----------
        path : str
            The qml file path
        """
        
        # parse an xml file by label
        mydoc = minidom.parse(path)
        items = mydoc.getElementsByTagName('paletteEntry')
        
        self.states = [State(elem.attributes['label'].value,
                             int(elem.attributes['value'].value),
                             elem.attributes['color'].value) for elem in items]
        
        return(self)
    
    def get_list_of_labels_values_colors(self):
        """
        Get the labels list, the values list and the colors list.

        Returns
        -------
        labels : list of str
            The list of labels.
        values : list of int
            The list of values.
        colors : list of str
            The list of colors.

        """
        labels = [state.label for state in self.states]
        values = [state.value for state in self.states]
        colors = [state.color for state in self.states]
        
        return(labels, values, colors)
    
    def sort(self, inplace=True):
        """
        Sort the palette according states values.

        Parameters
        ----------
        inplace : bool, default=False
            If True, perform operation in-place.

        Returns
        -------
        palette : Palette
            If ``inplace=True``, return the self object, else return a new palette with sorted states.

        """
        _, values, _ = self.get_list_of_labels_values_colors()
        
        index_sorted = list(np.argsort(values))
        
        ordered_states = [self.states[i] for i in index_sorted]
        
        if inplace:
            return(Palette(ordered_states))
        else:
            self.states = ordered_states
            return(self)