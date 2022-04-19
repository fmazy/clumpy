#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:03:33 2021

@author: frem
"""

import numpy as np

from .._base import Palette
from ..tools._console import title_heading

# READ ME
# Transition Probability Estimators (TPE) must have these methods :
#   - fit()
#   - transition_probability()
#   - _check()

class TransitionProbabilityEstimator():
    """
    Transition probability estimator base class.
    """
    def __init__(self,
                 n_corrections_max=1000,
                 log_computations=False,
                 verbose=0,
                 verbose_heading_level=1):
        self.verbose = verbose
        self.verbose_heading_level = verbose_heading_level

        self.n_corrections_max = n_corrections_max
        self.log_computations = log_computations

        # list of fitted final states.
        self._palette_fitted_states = Palette()


    

    def transition_probabilities(self,
                                 transition_matrix,
                                 Y,
                                 id_J=None,
                                 compute_P_Y__v=True,
                                 compute_P_Y = True,
                                 save_P_Y__v=False,
                                 save_P_Y=False):
        """
        Estimates transition probability. Non estimated final states transition probabilities are filled to the null value.

        Parameters
        ----------
        transition_matrix : TransitionMatrix
            Land transition matrix with only one state in ``tm.palette_u``.

        Y : array-like of shape (n_samples, n_features)
            Samples to estimate transition probabilities

        Returns
        -------
        P_v__Y : ndarray of shape (n_samples, n_transitions)
            Transition probabilities for each studied final land use states ordered
            as ``transition_matrix.palette_v``.
        """

        if id_J is None:
            id_J = np.ones(Y.shape[0]).astype(bool)

        # check if it is really a land transition matrix
        transition_matrix._check_land_transition_matrix()

        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level) + 'TPE computing')

        # P(Y) estimation
        if compute_P_Y:
            P_Y = self._compute_P_Y(Y=Y[id_J])
        else:
            P_Y = self.P_Y[id_J]

        # P(Y|v) estimation
        if compute_P_Y__v:
            list_v = transition_matrix.palette_v.get_list_of_values()
            P_Y__v = self._compute_P_Y__v(Y=Y[id_J],
                                          list_v=list_v)
        else:
            P_Y__v = self.P_Y__v[id_J]

        # BAYES ADJUSTMENT PROCESS
        P_v__Y = self._bayes_adjustment(P_Y__v=P_Y__v,
                                        P_Y=P_Y,
                                        transition_matrix=transition_matrix)

        if self.verbose > 0:
            print('TPE computing done.')

        if save_P_Y__v:
            self.P_Y__v = P_Y__v

        if save_P_Y:
            self.P_Y = P_Y

        return (P_v__Y)

    def _bayes_adjustment(self,
                          P_Y__v,
                          P_Y,
                          transition_matrix):

        P_v = transition_matrix.M[0,:]

        if self.log_computations == False:
            # if no log computation
            idx_not_null = P_Y[:,0] > 0
            P_v__Y = np.zeros_like(P_Y__v)
            P_v__Y[idx_not_null] = P_Y__v[idx_not_null] / P_Y[idx_not_null]

            P_v__Y_mean = P_v__Y.mean(axis=0)
            multiply_cols = P_v__Y_mean != 0
            P_v__Y[:, multiply_cols] *= P_v[multiply_cols] / P_v__Y_mean[multiply_cols]

            # P_v__Y *= P_v / P_v__Y.mean(axis=0)

            s = P_v__Y.sum(axis=1)

        else:
            # with log computation
            log_P_Y__v = np.zeros_like(P_Y__v)
            log_P_Y__v.fill(-np.inf)

            log_P_Y__v = np.log(P_Y__v, where=P_Y__v > 0, out=log_P_Y__v)

            log_P_Y = np.log(P_Y)

            log_P_v__Y = log_P_Y__v - log_P_Y
            log_P_v__Y -= np.log(np.mean(np.exp(log_P_v__Y), axis=0))
            log_P_v__Y += np.log(P_v)

            s = np.sum(np.exp(log_P_v__Y), axis=1)

        if np.sum(s > 1) > 0:
            if self.verbose > 0:
                s_sum = np.sum(s>1)
                print('Warning, '+str(s_sum)+'/'+str(s.size)+' ('+str(np.round(s_sum/s.size*100,2))+' %) uncorrect probabilities have been detected.')
                print('Some global probabilities may be to high.')
                print('For now, some corrections are made.')

            n_corrections = 0

            while np.sum(s > 1) > 0 and n_corrections < self.n_corrections_max:
                id_anomalies = s > 1

                if self.log_computations == False:
                    # if no log computation
                    P_v__Y[id_anomalies] = P_v__Y[id_anomalies] / \
                                           s[id_anomalies][:, None]

                    P_v__Y_mean = P_v__Y.mean(axis=0)
                    multiply_cols = P_v__Y_mean != 0
                    P_v__Y[:, multiply_cols] *= P_v[multiply_cols] / P_v__Y_mean[multiply_cols]

                    # P_v__Y *= P_v / P_v__Y.mean(axis=0)

                    s = np.sum(P_v__Y, axis=1)
                else:
                    # with log computation
                    log_P_v__Y[id_anomalies] = log_P_v__Y[id_anomalies] - np.log(s[id_anomalies][:, None])

                    log_P_v__Y -= np.log(np.mean(np.exp(log_P_v__Y), axis=0))
                    log_P_v__Y += np.log(P_v)

                    s = np.sum(np.exp(log_P_v__Y), axis=1)

                n_corrections += 1

            if self.verbose > 0:
                print('Corrections done in ' + str(n_corrections) + ' iterations.')

            if n_corrections == self.n_corrections_max:
                print(
                    'Warning : the P(v|Y) adjustment algorithm has reached the maximum number of loops. The n_corrections_max parameter should be increased.')

        if self.log_computations:
            P_v__Y = np.exp(log_P_v__Y)

        # last control to ensure s <= 1
        id_anomalies = s > 1
        P_v__Y[id_anomalies] = P_v__Y[id_anomalies] / \
                               s[id_anomalies][:, None]

        # avoid nan values
        P_v__Y = np.nan_to_num(P_v__Y)

        # compute the non transited column
        state_u = transition_matrix.palette_u.states[0]
        id_state_u = transition_matrix.palette_v.get_id(state_u)
        P_v__Y[:, id_state_u] = 1 - np.delete(P_v__Y, id_state_u, axis=1).sum(axis=1)

        return(P_v__Y)

