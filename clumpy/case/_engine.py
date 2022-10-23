# -*- coding: utf-8 -*-

from ..transition_probability_estimation import _methods as transition_probability_estimation_methods
from ..ev_selection import _methods as ev_selection_methods
from ..ev_selection import EVSelectors
from ..allocation import _methods as allocation_methods
from ..patch import _methods as patch_methods

class Engine():
    def __init__(self, 
                 initial_value,
                 final_values,
                 transition_probability_estimator=None,
                 ev_selectors=None,
                 allocator=None,
                 patcher=None):
        
        self.initial_value = initial_value
        self.final_values = final_values
        
        
        if type(transition_probability_estimator) is str:
            self.transition_probability_estimator = transition_probability_estimation_methods[transition_probability_estimator]()
        else:
            self.transition_probability_estimator = transition_probability_estimator
        
        if type(ev_selectors) is str:
            self.ev_selectors = EVSelectors(selectors={v:ev_selection_methods[ev_selectors]() for v in final_values})
            
        else:
            self.ev_selectors = ev_selectors
        
        if type(allocator) is str:
            self.allocator = allocation_methods[allocator]()
        else:
            self.allocator = allocator 
        
        if type(patcher) is str:
            self.patcher = [patch_methods[patcher]() for v in final_values]
        else:
            self.patcher = patcher
    
    