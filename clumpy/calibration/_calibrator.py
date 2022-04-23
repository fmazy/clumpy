# -*- coding: utf-8 -*-
from time import time
from copy import deepcopy
import numpy as np
from scipy import ndimage

from ..layer import Layer, FeatureLayer, LandUseLayer, MaskLayer
from .._base import State
from ..feature_selection import Pipeline
from ..patch import Patcher, Patchers

class Calibrator():
    def __init__(self,
                 state,
                 final_states,
                 tpe,
                 feature_selector=None,
                 patchers=None,
                 verbose = 0):
        self.state = state
        self.final_states = final_states
        self.tpe = tpe
        
        if feature_selector is None:
            self.feature_selector = Pipeline(fs_list=[])
        else:
            self.feature_selector = feature_selector
        
        self._fitted = False
        
        
        if isinstance(patchers, Patcher):
            self.patchers = Patchers()
            for v in final_states:
                if int(v) != int(self.state):
                    self.patchers[v] = patchers.copy()
        else:
            self.patchers = patchers
        
        self.verbose = verbose
        
    def __repr__(self):
        return ('Calibrator(tpe='+str(self.tpe)+')')
    
    def copy(self):
        return(Calibrator(state=deepcopy(self.state),
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
            mask=None,
            distances_to_states={}):
        """
        """
        
        self.features = features
        
        J, V = self.get_J_V(lul_initial=lul_initial,
                            lul_final=lul_final,
                            final_states_only=True)
                                        
        if self.verbose > 0:
            print('feature selecting...')
        
        # get X
        X = self.get_X(J=J,
                       features=self.features,
                       lul=lul_initial,
                       distances_to_states=distances_to_states,
                       selected_features=False)
        
        X = self.feature_selector.fit_transform(X, V)
        
        if self.verbose > 0:
            print('feature selecting done.')
        
        # BOUNDARIES PARAMETERS
        bounds = self.feature_selector.get_bounds(features=self.features)
                
        # TRANSITION PROBABILITY ESTIMATOR
        self.tpe.fit(X=X,
                     V=V,
                     initial_state = int(self.state),
                     bounds = bounds)
        
        self._fitted = True
        
        if self.patchers is not None:
            self.patchers.fit(J=J,
                              V=V,
                              shape=lul_initial.get_data().shape)
        
        return(self)
    
    def transition_probabilities(self,
                                 lul,
                                 tm,
                                 features=None,
                                 mask=None,
                                 distances_to_states={},
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
        
        if features is None:
            features = self.features
        
        tm = tm.extract(self.state)
        
        J = self.get_J(lul=lul,
                       mask=mask)
                
        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level) + 'Land ' + str(state) + ' TPE\n')

        # GET VALUES
        Y = self.get_X(J=J,
                       features=features,
                       lul=lul,
                       distances_to_states=distances_to_states)
        
        
        
        # TRANSITION PROBABILITY ESTIMATION
        P_v = tm.M[0,:]
        
        P_v__u_Y, final_states = self.tpe.transition_probabilities(
            J=J,
            Y=Y,
            P_v=P_v)
        
        if effective_transitions_only:
            bands_to_keep = np.array(self.final_states) != int(self.state)
            final_states = list(np.array(self.final_states)[bands_to_keep])
            P_v__u_Y = P_v__u_Y[:, bands_to_keep]
        
        if self.verbose > 0:
            print('Land ' + str(state) + ' TPE done.\n')

        if return_Y:
            return J, P_v__u_Y, final_states, Y
        else:
            return J, P_v__u_Y, final_states
    
    def get_J(self,
              lul,
              mask=None):
        """
        Get J indices.
    
        Parameters
        ----------
        lul : {'initial', 'final', 'start'} or LandUseLayer or np.array
            The land use map.
        mask : {'calibration', 'allocation'} or MaskLayer or np.array
            The mask.
    
        Returns
        -------
        None.
    
        """
                        
        # initial data
        # the region is selected after the distance computation
        if isinstance(lul, LandUseLayer):
            data_lul = lul.get_data().copy()
        else:
            data_lul = lul.copy()
    
        # selection according to the region.
        # one set -1 to non studied data
        # -1 is a forbiden state value.
        if mask is not None:
            if isinstance(mask, MaskLayer):
                data_lul[mask.get_data() != 1] = -1
            else:
                data_lul[mask != 1] = -1
    
        # get pixels indexes whose initial states are u
        return(np.where(data_lul.flat == int(self.state))[0])

    def get_V(self,
              lul,
              J,
              final_states_only=True):
                
        if isinstance(lul, LandUseLayer):
            data_lul = lul.get_data().copy()
        else:
            data_lul = lul.copy()
        
        V = data_lul.flat[J]
            
        if final_states_only:
            V[~np.isin(V, self.final_states)] = int(self.state)
        
        return(V)
    
    def get_J_V(self,
                lul_initial,
                lul_final,
                mask=None,
                final_states_only=True):
        J = self.get_J(lul=lul_initial,
                  mask=mask)
        V = self.get_V(lul=lul_final,
                  J=J,
                  final_states_only=final_states_only)
        return(J, V)
    
    def get_X(self, 
              J,
              features,
              lul,
              distances_to_states={},
              selected_features=True):
        
        X = None
    
        if selected_features:
            features = self.get_selected_features(features=features)
        
        for info in features:
            # switch according z_type
            if isinstance(info, Layer):
                # just get data
                x = info.get_data().flat[J]

            elif isinstance(info, int):
                # get distance data
                if info not in distances_to_states.keys():
                    _compute_distance(info, lul.get_data(), distances_to_states)
                    
                x = distances_to_states[info].flat[J]
                
            else:
                logger.error('Unexpected feature info : ' + type(info) + '. Occured in \'_base/_land.py, Land.get_values()\'.')
                raise (TypeError('Unexpected feature info : ' + type(info) + '.'))

            # if X is not yet defined
            if X is None:
                X = x
            # else column stack
            else:
                X = np.column_stack((X, x))

        # if only one feature, reshape X as a column
        if len(X.shape) == 1:
            X = X[:, None]
        
        return(X)
    
    def compute_bootstrap_patches(self,
                                  lul_initial,
                                  lul_final,
                                  mask=None):
        """
        Compute Bootstrap patches

        """
        J, V = self.get_J_V(lul_initial = lul_initial,
                            lul_final = lul_final,
                            mask=mask,
                            final_states_only=False)
        
        self.patches = compute_bootstrap_patches(state=self.state,
                                            final_states=self.final_states,
                                            J=J,
                                            V=V,
                                            shape=shape,
                                            mask=mask)       
        

def _compute_distance(state_value, data, distances_to_states):
    v_matrix = (data == state_value).astype(int)
    distances_to_states[state_value] = ndimage.distance_transform_edt(1 - v_matrix)