# -*- coding: utf-8 -*-

import numpy as np

class ExpertSelector():
    def __init__(self, 
                 select=[],
                 ev_labels=None,
                 region_label=None,
                 initial_state=None,
                 final_state=None):
        self.select = select
        self._cols_support = np.array(self.select)
        self.region_label=region_label,
        self.initial_state=initial_state,
        self.final_state=final_state
        
        def __repr__(self):
            return 'ExpertSelector()'
        
        def set_params(self, **params):
            """
            Set parameters.

            Parameters
            ----------
            **params : kwargs
                Parameters et values to set.

            Returns
            -------
            self : CramerMRMR
                The self object.

            """
            for param, value in params.items():
                setattr(self, param, value)
                
        def fit(self, **kwargs):
            
            return(self)