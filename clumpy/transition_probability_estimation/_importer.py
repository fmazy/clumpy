# -*- coding: utf-8 -*-
import numpy as np

from ._tpe import TransitionProbabilityEstimator
from ..layer import ProbaLayer

class Importer(TransitionProbabilityEstimator):
    def __init__(self,
                 initial_state,
                 proba_layer,
                 verbose=0,
                 verbose_heading_level=1,
                 **kwargs):

        super().__init__(initial_state,
                         verbose=verbose,
                         verbose_heading_level=verbose_heading_level)
        
        self.proba_layer = proba_layer
        self.yielder = self.proba_layer.yield_proba_of_initial_state(initial_state=self.initial_state)
        
    def fit(self,
            **kwargs):
        
        return self
        
    def transition_probabilities(self, 
                                 J, 
                                 **kwargs):
        P_v__u_Y = np.array([]).reshape((J.size, 0))
        
        final_states = []
        
        for final_state, P in self.yielder:
            P_v__u_Y = np.hstack((P_v__u_Y, P.flat[J][:,None]))
            
            final_states.append(final_state)
        
        if self.initial_state not in final_states:
            P_v__u_Y = np.hstack((P_v__u_Y, 1-P_v__u_Y.sum(axis=1)))
            final_states.append(self.initial_state)
        
        return(P_v__u_Y, final_states)