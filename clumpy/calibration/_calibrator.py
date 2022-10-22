# -*- coding: utf-8 -*-
from time import time
from copy import deepcopy
import numpy as np
from scipy import ndimage

from ..layer import Layer, FeatureLayer, LandUseLayer, MaskLayer
from .._base import State
from ..feature_selection import FeatureSelectors
from ..patch import Patcher, Patchers

class Calibrator():
    def __init__(self,
                 initial_state,
                 final_states,
                 transition_probability_estimator,
                 ev_selector=None,
                 patchers=None,
                 verbose = 0):
        if initial_state in final_states:
            final_states.remove(initial_state)
        
        self.initial_state = initial_state
        self.final_states = final_states
        
        self.transition_probability_estimator = transition_probability_estimator
        
        if ev_selector is None:
            self.ev_selector = FeatureSelectors()
        else:
            self.ev_selector = ev_selector
        
        self._fitted = False
        
        
        if isinstance(patchers, Patcher):
            self.patchers = Patchers()
            for v in final_states:
                if int(v) != int(self.initial_state):
                    self.patchers[v] = patchers.copy()
        else:
            self.patchers = patchers
        
        self.verbose = verbose
        
    def __repr__(self):
        return ('Calibrator(tpe='+str(self.tpe)+')')
    
    def copy(self):
        return(Calibrator(initial_state=deepcopy(self.initial_state),
                          tpe=deepcopy(self.tpe),
                          feature_selector=self.feature_selector.copy(),
                          verbose=self.verbose))
    
    def check(self, objects=None):
        """
        Check the unicity of objects.
        Notably, estimators uniqueness are checked to avoid malfunctioning during transition probabilities estimation.
        """
        if objects is None:
            objects = []
            
        if self.feature_selector in objects:
            raise(ValueError("Features objects must be different."))
        else:
            objects.append(self.feature_selector)
        
        if isinstance(self.feature_selector, Pipeline):
            self.feature_selector.check(objects=objects)
        
        if self.tpe in objects:
            raise(ValueError("TPE objects must be different."))
        else:
            objects.append(self.tpe)
        
        self.tpe.check(objects=objects)
        
        if self.patchers is not None and self.patchers in objects:
            raise(ValueError("Patchers objects must be different."))
        else:
            objects.append(self.patchers)
        
        self.patchers.check(objects=objects)
        
    
    def fit(self,
            lul_initial,
            lul_final,
            features,
            mask=None):
        """
        """
        if len(self.final_states) == 0 or self.final_states == [self.initial_state]:
            return self
        
        self.features = features
        
        J = lul_initial.get_J(state = self.initial_state,
                              mask = mask)
        
        J, V = lul_final.get_V(J=J,
                               final_states=self.final_states + [self.initial_state])
                                                
        if self.verbose > 0:
            print('feature selecting...')
        
        # get X
        X = lul_initial.get_X(J=J,
                              features=self.features)
        
        X = self.feature_selector.fit_transform(X, V)
                
        if self.verbose > 0:
            print('feature selecting done.')
        
        # BOUNDARIES PARAMETERS
        bounds = self.feature_selector.get_bounds(features=self.features)
                
        # TRANSITION PROBABILITY ESTIMATOR
        self.tpe.fit(X=X,
                     V=V,
                     initial_state = int(self.initial_state),
                     bounds = bounds)
        
        self._fitted = True
        
        if self.patchers is not None:
            self.patchers.fit(J=J,
                              V=V,
                              shape=lul_initial.shape)
        
        return(self)
    
    def transition_probabilities(self,
                                 lul,
                                 tm,
                                 features=None,
                                 mask=None,
                                 effective_transitions_only=True,
                                 return_Y=False):
        """
        Computes transition probabilities.

        Parameters
        ----------
        transition_matrix : TransitionMatrix
            Land transition matrix with only one state in ``tm.palette_u``.

        lul : LandUseLayer
            The studied land use layer.

        mask : MaskLayer, default = None
            The region mask layer. If ``None``, the whole area is studied.

        distances_to_states : dict(State:ndarray), default={}
            The distances matrix to key state. Used to improve performance.

        path_prefix : str, default=None
            The path prefix to save result as ``path_prefix+'_'+str(state_v.value)+'.tif'.
            Note that if ``path_prefix is not None``, ``lul`` must be LandUseLayer

        save_P_Y__v : bool, default=False
            Save P_Y__v.

        save_P_Y : bool, default=False
            Save P_Y

        return_Y : bool, default=False
            If ``True`` and ``path_prefix`` is ``None``, return Y.

        Returns
        -------
        J_allocation : ndarray of shape (n_samples,)
            Element indexes in the flattened
            matrix.

        P_v__u_Y : ndarray of shape (n_samples, len(palette_v))
            The transition probabilities of each elements. Columns are
            ordered as ``palette_v``.

        Y : ndarray of shape (n_samples, n_features)
            Only returned if ``return_Y=True``.
            The features values.
        """
        
        if self.verbose>0:
            print('Calibrator transition probabilities estimation')
        
        if features is None:
            features = self.features
                
        J = lul.get_J(state = self.initial_state,
                      mask = mask)
                
        # GET VALUES
        Y = lul.get_X(J=J,
                      features=features)
        
        Y = self.feature_selector.transform(Y)
        
        # TRANSITION PROBABILITY ESTIMATION
        P_v = np.array([tm.get(self.initial_state, final_state) for final_state in self.final_states])
        
        P_v__u_Y, final_states = self.tpe.transition_probabilities(
            J=J,
            Y=Y,
            P_v=P_v)
        
        if effective_transitions_only:
            bands_to_keep = np.array(self.final_states) != int(self.initial_state)
            final_states = list(np.array(self.final_states)[bands_to_keep])
            P_v__u_Y = P_v__u_Y[:, bands_to_keep]
        
        if return_Y:
            return J, P_v__u_Y, final_states, Y
        else:
            return J, P_v__u_Y, final_states
    
    # def get_J(self,
    #           lul,
    #           mask=None):
    #     """
    #     Get J indices.
    
    #     Parameters
    #     ----------
    #     lul : {'initial', 'final', 'start'} or LandUseLayer or np.array
    #         The land use map.
    #     mask : {'calibration', 'allocation'} or MaskLayer or np.array
    #         The mask.
    
    #     Returns
    #     -------
    #     None.
    
    #     """
                        
    #     # initial data
    #     # the region is selected after the distance computation
    #     if isinstance(lul, LandUseLayer):
    #         data_lul = lul.get_data().copy()
    #     else:
    #         data_lul = lul.copy()
    
    #     # selection according to the region.
    #     # one set -1 to non studied data
    #     # -1 is a forbiden state value.
    #     if mask is not None:
    #         if isinstance(mask, MaskLayer):
    #             data_lul[mask.get_data() != 1] = -1
    #         else:
    #             data_lul[mask != 1] = -1
    
    #     # get pixels indexes whose initial states are u
    #     return(np.where(data_lul.flat == int(self.initial_state))[0])

    # def get_V(self,
    #           lul,
    #           J,
    #           final_states_only=True):
                
    #     if isinstance(lul, LandUseLayer):
    #         data_lul = lul.get_data().copy()
    #     else:
    #         data_lul = lul.copy()
        
    #     V = data_lul.flat[J]
            
    #     if final_states_only:
    #         V[~np.isin(V, self.final_states)] = int(self.initial_state)
        
    #     return(V)
    
    # def get_J_V(self,
    #             lul_initial,
    #             lul_final,
    #             mask=None,
    #             final_states_only=True):
    #     J = self.get_J(lul=lul_initial,
    #               mask=mask)
    #     V = self.get_V(lul=lul_final,
    #               J=J,
    #               final_states_only=final_states_only)
    #     return(J, V)
    
    # def get_X(self, 
    #           J,
    #           features,
    #           lul,
    #           distances_to_states={},
    #           selected_features=True):
        
    #     X = None
    
    #     if selected_features:
    #         features = self.get_selected_features(features=features)
        
    #     for info in features:
    #         # switch according z_type
    #         if isinstance(info, Layer):
    #             # just get data
    #             x = info.get_data().flat[J]

    #         elif isinstance(info, int):
    #             # get distance data
    #             if info not in distances_to_states.keys():
    #                 _compute_distance(info, lul.get_data(), distances_to_states)
                    
    #             x = distances_to_states[info].flat[J]
                
    #         else:
    #             logger.error('Unexpected feature info : ' + type(info) + '. Occured in \'_base/_land.py, Land.get_values()\'.')
    #             raise (TypeError('Unexpected feature info : ' + type(info) + '.'))

    #         # if X is not yet defined
    #         if X is None:
    #             X = x
    #         # else column stack
    #         else:
    #             X = np.column_stack((X, x))

    #     # if only one feature, reshape X as a column
    #     if len(X.shape) == 1:
    #         X = X[:, None]
        
    #     return(X)