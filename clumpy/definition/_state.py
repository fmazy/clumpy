#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 22:39:48 2021

@author: frem
"""

from xml.dom import minidom

class State():
    def __init__(self, name, value, color):
        self.name = name
        self.value = value
        self.color = color
    
    def __repr__(self):
        return(self.name)
        
class Palette():
    def __init__(self):
        self.states = []
    
    def __repr__(self):
        return(str(self.states))
        
    def add_state(self, state):
        for s in self.states:
            if s.value == state.value:
                raise(ValueError('Unexpected state : its value is already set'))
    
        self.states.append(state)
        
        return(self)
    
    def remove_state_by_value(self, value):
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
        
        # parse an xml file by name
        mydoc = minidom.parse(path)
        items = mydoc.getElementsByTagName('paletteEntry')
        
        self.states = [State(elem.attributes['label'].value,
                             elem.attributes['value'].value,
                             elem.attributes['color'].value) for elem in items]
        
        return(self)