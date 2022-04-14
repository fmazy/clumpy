# -*- coding: utf-8 -*-

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