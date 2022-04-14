# -*- coding: utf-8 -*-
from copy import deepcopy

class Features():
    def __init__(self, 
                 list = [],
                 selector = None):
        self.list = list
        self.selector = selector
    
    def __repr__(self):
        return str(self.list)+', '+str(self.selector)
    
    def __iter__(self):
        for feature in self.list:
            yield feature

    def __len__(self):
        return (len(self.list))
    
    def __getitem__(self, i):
         return self.list[i]
    
    def copy(self):
        return(Features(list = [f for f in self.list],
                        selector = deepcopy(self.selector)))