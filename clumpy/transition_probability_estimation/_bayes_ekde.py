# -*- coding: utf-8 -*-

from . import Bayes

class BayesEKDE(Bayes):
    def __init__(self,
                 n_corrections_max=1000,
                 n_fit_max=10**5,
                 log_computations=False,
                 P_Y__v_layer=None,
                 verbose=0,
                 verbose_heading_level=1,
                 **kwargs):

        super().__init__(density_estimator='ekde',
                         n_corrections_max=n_corrections_max,
                         n_fit_max=n_fit_max,
                         log_computations=log_computations,
                         P_Y__v_layer=P_Y__v_layer,
                         verbose=verbose,
                         verbose_heading_level=verbose_heading_level)
        