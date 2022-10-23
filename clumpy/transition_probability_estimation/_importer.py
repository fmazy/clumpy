# -*- coding: utf-8 -*-
import numpy as np

from ._tpe import TransitionProbabilityEstimator
from ..layer import ProbaLayer

class Importer(TransitionProbabilityEstimator):
    def __init__(self,
                 proba:ProbaLayer,
                 verbose=0,
                 verbose_heading_level=1,
                 **kwargs):

        super().__init__(verbose=verbose,
                         verbose_heading_level=verbose_heading_level)
        
        self.proba = proba
        
    def fit(self,
            **kwargs):
        
        return self
    
    def transition_probabilities(self, 
                                 J, 
                                 **kwargs):
        P_v__u_Y = np.array([]).reshape((J.size, 0))
        
        final_states = list(self.get_final_states())
        for i, final_state in enumerate(final_states):
            P = self.proba[i,:,:]
            P_v__u_Y = np.hstack((P_v__u_Y, P.flat[J][:,None]))
            
        if self.initial_state not in final_states:
            P_v__u_Y = np.hstack((P_v__u_Y, 1-P_v__u_Y.sum(axis=1)[:,None]))
            final_states.append(self.initial_state)
        
        return(P_v__u_Y, final_states)
    
    def get_final_states(self):
        return self.proba.final_states