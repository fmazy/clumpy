#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
State blabla.
"""

from xml.dom import minidom
import numpy as np

import logging
logger = logging.getLogger('clumpy')

from ..tools._console import stop_log

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
        if value < 0:
            logger.error('Unexpected state value. It must be positive !')
            stop_log()
            raise

        self.label = label
        self.value = value
        self.color = color

    def __repr__(self):
        return (self.label)
    
    def __int__(self):
        return(self.value)


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
            _check_states(states)

            self.states = states

    def __repr__(self):
        return (str(self.states))

    def __iter__(self):
        for state in self.states:
            yield state

    def __len__(self):
        return (len(self.states))
    
    def __getitem__(self, info):
         return self.get(info)

    def copy(self):
        palette = Palette(states = [state for state in self.states])
        return(palette)
    
    def print_table(self, colors=False):
        for state in self.states:
            if colors:
                print(state.value, state.label, state.color)
            else:
                print(state.value, state.label)

    def add(self, state):
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
        if state.value in self.get_list_of_values():
            logger.error('Unexpected state : its value is already set')
            stop_log()
            raise 

        self.states.append(state)

        return (self)

    def get_id(self, info):
        """
        Get a state's index from the palette.
        
        Parameters
        ----------
        info : State or int or str
            The state information which can be the object, the state's value or the state's label.
            If two states share the same label, only the first one to occur is returned.

        Returns
        -------
        i : int
            The requested state's index.

        """
        if isinstance(info, State):
            return (self._get_id(info))
        elif isinstance(info, int):
            return (self._get_id_by_value(info))
        elif isinstance(info, str):
            return (self._get_id_by_label(info))

    def get(self, info):
        """
        Get a state from the palette.

        Parameters
        ----------
        info : int or str
            The state information which can be the state's value or the state's label.
            If two states share the same label, only the first one to occur is returned.

        Returns
        -------
        state : State
            The requested state.
        """
        if isinstance(info, int) or isinstance(info, np.integer):
            return (self._get_by_value(info))
        elif isinstance(info, str):
            return (self._get_by_label(info))
        elif isinstance(info, State):
            return (info)
        else:
            logger.error("Unexpected info type. Should be int or str or State")
            stop_log()
            raise 

    def extract(self, infos):
        """
        Extract a sub palette

        Parameters
        ----------
        infos : list(int or State)
            List of initial state information which can be the object, the state's value or the state's label.

        Returns
        -------
        palette : Palette
            The extracted palette.
        """
        states = [self.get(info) for info in infos]

        _check_states(states)

        return (Palette(states=states))

    def _get_id(self, state):
        """
        Get a state index.
        
        Parameters
        ----------
        state : State
            The state object.
        
        Returns
        -------
        i : int
            The requested index.
        """
        return (self.states.index(state))

    def _get_by_value(self, value):
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
        
        try:
            return (self.states[values.index(value)])
        except:
            logger.error(str(value) +" is not in the palette. Occured in '_base/_state.py, Palette._get_by_value()'.")
            # stop_log()
            raise

    def _get_id_by_value(self, value):
        """
        Get a state's id by its value.

        Parameters
        ----------
        value : int
            The researched value.

        Returns
        -------
        i : int
            The requested state's id'
        """
        values = [state.value for state in self.states]

        return (values.index(value))

    def _get_by_label(self, label):
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

        return (self.states[labels.index(label)])

    def _get_id_by_label(self, label):
        """
        Get a state's id by its label.
        If two states have the same label, only the first one is returned.

        Parameters
        ----------
        label : int
            The researched label.

        Returns
        -------
        i : int
            The requested state's id.
        """
        labels = [state.label for state in self.states]

        return (labels.index(label))

    def merge(self, palette, inplace=False):
        """
        Merge to another palette.

        Parameters
        ----------
        palette : Palette
            The other palette

        inplace : bool, default=False
            If True, perform operation in-place.

        Returns
        -------
        palette : Palette
            If ``inplace=True``, return the self object, else return a new palette with sorted states.
        """
        states = [state for state in self.states]

        for state in palette.states:
            if state not in states:
                states.append(state)

        merged_palette = Palette(states).sort(inplace=True)

        if inplace:
            self.states = merged_palette.states
        else:
            return (merged_palette)

    def remove(self, info, inplace=False):
        """
        Remove a state

        Parameters
        ----------
        info : State or int or str
            The state information which can be the object, the state's value or the state's label.
            If two states share the same label, only the first one to occur is returned.

        inplace : bool, default=False
            If True, perform operation in-place.

        Returns
        -------
        palette : Palette
            If ``inplace=True``, return the self object, else return a new palette with sorted states.

        """

        if inplace:
            self.states.remove(self.get(info))
            return (self)

        else:
            states = [state for state in self.states]
            states.remove(self.get(info))

            return (Palette(states))

    def get_list_of_values(self):
        """
        Get the values list.

        Returns
        -------
        values : list of int
            The list of values.

        """
        return ([state.value for state in self.states])

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

        return (labels, values, colors)

    def sort(self, inplace=False):
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
            self.states = ordered_states
            return (self)
        else:
            return (Palette(ordered_states))


def _check_states(states):
    """
    Check if they are duplicates in states (object and state values).
    """
    if len(set(states)) != len(states):
        logger.error('Duplicate states found !')
        stop_log()
        raise

    values = []
    for state in states:
        if state.value in values:
            stop_log()
            logger.error('Duplicate states values found !')
            raise
        values.append(state.value)


def load_palette(path):
    """
    Import legend through qml file provided by QGis. Alpha is not supported.

    Parameters
    ----------
    path : str
        The qml file path

    Returns
    -------
    palette : Palette
        The loaded palette.
    """

    # parse an xml file by label
    mydoc = minidom.parse(path)
    items = mydoc.getElementsByTagName('paletteEntry')

    states = [State(elem.attributes['label'].value,
                    int(elem.attributes['value'].value),
                    elem.attributes['color'].value) for elem in items]

    _check_states(states=states)

    return (Palette(states=states))
