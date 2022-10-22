# -*- coding: utf-8 -*-

from ..transition_probability_estimation import _methods as transition_probability_estimation_methods

class Engine():
    def __init__(self, 
                 initial_value,
                 final_value,
                 transition_probability_estimator=None,
                 ev_selector=None,
                 allocator=None,
                 patcher=None):
        
        self.initial_value = initial_value
        self.final_values = final_value
        
        
        if type(transition_probability_estimator) is str:
            transition_probability_estimation_methods[transition_probability_estimator]
        else:
            self.transition_probability_estimator = transition_probability_estimator
        
        
        self.ev_selector = ev_selector
        self.allocator = allocator
        self.patcher = patcher
    
    