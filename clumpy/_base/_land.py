#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy import ndimage
import numpy as np

from ..density_estimation import _methods
from ..density_estimation._density_estimator import DensityEstimator
from ._layer import _Layer
from . import State
from ..estimation import TransitionProbabilityEstimator
from ..tools import path_split
from . import FeatureLayer, LandUseCoverLayer
from ..allocation._gart import generalized_allocation_rejection_test
from ..allocation._patcher import _weighted_neighbors


class Land():
    """
    Land object which refers to a given initial state.

    Parameters
    ----------
    features : list(FeaturesLayer) or list(State)
        List of features where a State means a distance layer to the corresponding state.
    de : {'gkde'} or DensityEstimator, default='gkde'
        Density estimation for :math:`P(x|u)`.
            
            gkde : Gaussian Kernel Density Estimation method
            
    update_P_v__u_Y : bool, default=True
        If ``True``, :math:`P(v|u,Y)` is updated at each loop during the
        allocation process.
    
    n_allocation_try : int, default=10**3
        The maximum number of allocation loops.
    
    verbose : int, default=None
        Verbosity level.
    """
    def __init__(self, 
                 features,
                 de = 'gkde',
                 update_P_v__u_Y = True,
                 n_allocation_try = 10**3,
                 verbose = 0):
        
        self.features = features
        
        if isinstance(de, DensityEstimator):
            self.density_estimator = de
        elif de in _methods:
            self.density_estimator = _methods[de]()
        else:
            raise(ValueError('Unexpected de value.'))
                
        self.conditional_density_estimators = {}
        self.patches = {}
        
        self.update_P_v__u_Y  = update_P_v__u_Y
        self.n_allocation_try = n_allocation_try
        
        self.verbose = verbose
        
    def __repr__(self):
        return('land')
    
    def add_conditional_density_estimator(self, state, de='gkde'):
        """
        Add conditional density for a given final state.

        Parameters
        ----------
        state : State
            The final state.
        de : {'gkde'} or DensityEstimator, default='gkde'
            Density estimation for :math:`P(x|u,v)`.
                gkde : Gaussian Kernel Density Estimation method
        Returns
        -------
        self : Land
            The self object.
        """
        
        if isinstance(de, DensityEstimator):
            self.conditional_density_estimators[state] = de
        elif de in _methods:
            self.conditional_density_estimators[state] = _methods[de]()
        else:
            raise(ValueError('Unexpected de value.'))
        
        return(self)
        
    def add_patch(self, 
                  state,
                  patch):
        """
        Add a patch object for a given final state.

        Parameters
        ----------
        state : State
            The final state.
        patch : Patch
            The corresponding Patch object.

        Returns
        -------
        self : Land
            The self object.
        """
        
        self.patches[state] = patch
        
        return(self)
    
    def get_values(self,
                   state,
                   luc_initial,
                   luc_final = None,
                   mask = None,
                   explanatory_variables=True,
                   distances_to_states={}):
        """
        Get values.

        Parameters
        ----------
        state : State
            The studied initial state.
            
        luc_initial : LandUseCoverLayer or ndarray
            The initial land use layer.
            
        luc_final : LandUseCoverLayer or ndarray, default=None
            The final land use layer. Ignored if ``None``.
            
        mask : LandUseCoverLayer, default=None
            The region mask layer. If ``None``, the whole area is studied.
            
        explanatory_variables : bool, default=True
            If ``True``, features values are returned.
            
        distances_to_states : dict(ndarray)
            A dict of ndarray distances_to_states to the State used as key.
            Usefull to avoid redondant distance computations.

        Returns
        -------
        J : ndarray of shape (n_samples,)
            The samples flat indexes.
        
        X : ndarray of shape (n_samples, n_features)
            Returned if ``explanatory_variables`` is ``True``. The features values.
        
        V : ndarray of shape (n_samples, n_features)
            Returned if ``luc_final`` is not ``None``. The final state values.

        """
        # initial data
        # the region is selected after the distance computation
        if isinstance(luc_initial, LandUseCoverLayer):
            data_luc_initial = luc_initial.get_data().copy()
        else:
            data_luc_initial = luc_initial.copy()
        
        # selection according to the region.
        # one set -1 to non studied data
        # -1 is a forbiden state value.
        if mask is not None:
            data_luc_initial[mask.get_data() != 1] = -1
        
        # get pixels indexes whose initial states are u
        # J = ndarray_suitable_integer_type(np.where(initial_luc_layer.raster_.read(1).flat==u)[0])
        J = np.where(data_luc_initial.flat == state.value)[0]
        
        X = None
        if explanatory_variables:
            # create feature labels
            for info in self.features:
                # switch according z_type
                if isinstance(info, _Layer):
                    # just get data
                    x = info.get_data().flat[J]
                
                elif isinstance(info, State):
                    # get distance data
                    # in this case, feature is a State object !
                    if info not in distances_to_states.keys():
                        _compute_distance(info, data_luc_initial, distances_to_states)
                    x = distances_to_states[info].flat[J]
                    
                else:
                    raise(TypeError('Unexpected feature.'))
                    
                # if X is not yet defined
                if X is None:
                    X = x
                
                # else column stack
                else:
                    X = np.column_stack((X, x))

            # if only one feature, reshape X as a column
            if len(self.features) == 1:
                X = X[:,None]
        
        # if final luc layer
        if luc_final is not None:
            if isinstance(luc_final, LandUseCoverLayer):
                data_luc_final = luc_final.get_data()
            else:
                data_luc_final = luc_final
            
            # just get data inside the region (because J is already inside)
            V = data_luc_final.flat[J]
        
    
        elements_to_return = [J]
    
        if explanatory_variables:
            elements_to_return.append(X)
        
        if luc_final is not None:
            elements_to_return.append(V)
        
        return(elements_to_return)
    
    def transition_probabilities(self,
                                 state,
                                 luc,
                                 P_v,
                                 palette_v,
                                 mask = None,
                                 distances_to_states={},
                                 path_prefix=None):
        """
        Computes transition probabilities.

        Parameters
        ----------
        state : State
            The initial state of this land.
            
        luc : LandUseCoverLayer
            The studied land use layer.
            
        P_v : ndarray of shape(len(palette_v,))
            The global transition probabilities :math:`P(v)`. The order corresponds to
            ``palette_v``
            
        palette_v : Palette
            The final state palette corresponding to ``P_v``.
        
        mask : LandUseCoverLayer, default = None
            The region mask layer. If ``None``, the whole area is studied.
        
        distances_to_states : dict(State:ndarray), default={}
            The distances matrix to key state. Used to improve performance.
        
        path_prefix : str, default=None
            The path prefix to save result as ``path_prefix+'_' + str(state.value) + '.tif'.
            If None, the result is returned.
            Note that if ``path_prefix is not None``, ``luc`` must be LandUseCoverLayer

        Returns
        -------
        J_allocation : ndarray of shape (n_samples,)
            Only returned if ``path_prefix=False``. Element indexes in the flattened
            matrix.
        
        P_v__u_Y : ndarray of shape (n_samples, len(palette_v))
            The transition probabilities of each elements. Columns are
            ordered as ``palette_v``.

        """
                
        J_allocation, P = self._compute_tpe(state=state,
                                             luc=luc,
                                             P_v = P_v,
                                             palette_v = palette_v,
                                             mask=mask,
                                             distances_to_states=distances_to_states)
        
        if path_prefix is None:
            return(J_allocation, P)
        
        else:
            print(path_prefix,path_split(path_prefix, prefix=True))
            folder_path, file_prefix = path_split(path_prefix, prefix=True)
            
            
            for id_state, state in enumerate(palette_v):
                M = np.zeros(luc.get_data().shape)
                M.flat[J_allocation] = P[:, id_state]
                
                file_name = file_prefix + '_' + str(state.value) + '.tif'
                
                FeatureLayer(label=file_name,
                                data = M,
                                copy_geo = luc,
                                path = folder_path + '/' + file_name)
            
            return(True)
    
    def fit(self,
            state,
            luc_initial,
            luc_final,
            mask=None,
            distances_to_states={}):
        """
        Fit the land. Required for any further process.

        Parameters
        ----------
        state : State
            The initial state of this land.
            
        luc_initial : LandUseCoverLayer
            The initial land use.
            
        luc_final : LandUseCoverLayer
            The final land use.
            
        mask : LandUseCoverLayer, default = None
            The region mask layer. If ``None``, the whole area is studied.
        
        distances_to_states : dict(State:ndarray), default={}
            The distances matrix to key state. Used to improve performance.

        Returns
        -------
        self : Land
            The self object.
        """
        
        self._fit_tpe(state=state,
                     luc_initial=luc_initial,
                     luc_final=luc_final,
                     mask=mask,
                     distances_to_states=distances_to_states)
        
        return(self)
    
    def _fit_tpe(self,
                 state,
                 luc_initial,
                 luc_final,
                 mask=None,
                 distances_to_states={}):
        """
        Fit the transition probability estimator
        """
        J_calibration, X, V = self.get_values(state = state,
                                luc_initial = luc_initial,
                                luc_final = luc_final,
                                mask = mask,
                                explanatory_variables=True,
                                distances_to_states=distances_to_states)
        
        self._tpe = TransitionProbabilityEstimator(density_estimator = self.density_estimator,
                                                   conditional_density_estimators = self.conditional_density_estimators)
        self._tpe.fit(X, V)
        
        return(self)
    
    def _compute_tpe(self,
                     state,
                     luc,
                     P_v,
                     palette_v,
                     mask=None,
                     distances_to_states={}):
        """
        Compute the transition probability estimation according to the given P_v
        """
        
        J_allocation, Y = self.get_values(state = state,
                                luc_initial = luc,
                                mask = mask,
                                explanatory_variables=True,
                                distances_to_states=distances_to_states)
                
        P = self._tpe.transition_probability(Y, P_v, palette_v)
        
        return(J_allocation, P)
    
    def allocation(self,
                   state,
                   P_v,
                   palette_v,
                   luc,
                   luc_origin=None,
                   mask=None,
                   distances_to_states={},
                   path=None):
        """
        allocation. luc can be both LandUseCoverLayer and ndarray.

        Parameters
        ----------
        state : TYPE
            DESCRIPTION.
        P_v : TYPE
            DESCRIPTION.
        palette_v : TYPE
            DESCRIPTION.
        luc : TYPE
            DESCRIPTION.
        luc_origin : TYPE, optional
            DESCRIPTION. The default is None.
        mask : TYPE, optional
            DESCRIPTION. The default is None.
        distances_to_states : TYPE, optional
            DESCRIPTION. The default is {}.
        path : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        if luc_origin is None:
            luc_origin = luc
        
        if isinstance(luc_origin, LandUseCoverLayer):
            luc_origin_data = luc_origin.get_data()
        else:
            luc_origin_data = luc_origin
            
        if isinstance(luc, LandUseCoverLayer):
            luc_data = luc.get_data().copy()
        else:
            luc_data = luc
        
        # plt.imshow(luc_data)
        # plt.show()
        
        # ghost initialization
        palette_v_without_u = palette_v.remove(state)
        n_ghost = {state_v : 1 for state_v in palette_v_without_u}
        
        # P_v initialization
        P_v = P_v.copy()
        
        # get the id of the initial state
        id_state = palette_v.get_id(state)
        
        # first_len_J initialization
        first_len_J = None
        
        J_used = []
        
        n_try = 0
        while np.sum(list(n_ghost.values())) > 0 and n_try < self.n_allocation_try:
            n_try += 1
            
            # P_v is divided by patch area mean
            # P_v_patches is then largely smaller.
            # one keep P_v to update it after the allocation try.
            P_v_patches = P_v.copy()
            P_v_patches[id_state] = 1
            for id_state_v, state_v in enumerate(palette_v):
                if state_v != state:
                    P_v_patches[id_state_v] /= self.patches[state_v].area_mean
                    P_v_patches[id_state] -= P_v[id_state_v]
            
            # if P_v__u_Y has to be updated
            # or if it is the first loop
            # compute P(v|u,Y)
            if self.update_P_v__u_Y or n_try == 1:
                luc_data_P_v__u_Y_update = luc_data.copy()
                luc_data_P_v__u_Y_update.flat[J_used] = -1
                J, P_v__u_Y = self._compute_tpe(state=state,
                                                luc=luc_data_P_v__u_Y_update,
                                                P_v = P_v_patches,
                                                palette_v = palette_v,
                                                mask = mask,
                                                distances_to_states=distances_to_states)
            else:
                # else just update J by removing used pixels.
                # it is faster but implies a bias.
                idx_to_keep = ~np.isin(J, J_used)
                J = J[idx_to_keep]
                P_v__u_Y = P_v__u_Y[idx_to_keep]
            
            # save the original number of pixels in this land
            # it is used to edit P(v)
            if first_len_J is None:
                first_len_J = len(J)
            
            # Allocation try
            # results are : all pixels used, number of allocation for each state
            # and numbre of ghost pixels for each states.
            J_used_last, n_allocated, n_ghost = self._try_allocation(state=state,
                                                J=J,
                                                P_v__u_Y=P_v__u_Y,
                                                palette_v=palette_v,
                                                luc_origin_data=luc_origin_data,
                                                luc_data = luc_data)
            
            # the pixel used and which are now useless are saved in J_used
            J_used += J_used_last
                       
            # P_v update : for the next loop, one need less pixels
            P_v[id_state] = 1
            for id_state_v, state_v in enumerate(palette_v):
                if state_v != state:
                    P_v[id_state_v] -= n_allocated[state_v] / first_len_J
                    P_v[id_state] -= P_v[id_state_v]
            
        
        if path is not None:
            folder_path, file_name, file_ext = path_split(path)
            return(LandUseCoverLayer(label = 'file_name',
                                     data = luc_data,
                                     copy_geo = luc_origin,
                                     path = path,
                                     palette = luc_origin.palette))
    
    def _try_allocation(self,
                     state,
                     J,
                     P_v__u_Y,
                     palette_v,
                     luc_origin_data,
                     luc_data):
        """
        Try to allocate

        Parameters
        ----------
        state : TYPE
            DESCRIPTION.
        J : TYPE
            DESCRIPTION.
        P_v__u_Y : TYPE
            DESCRIPTION.
        palette_v : TYPE
            DESCRIPTION.
        luc_origin_data : TYPE
            DESCRIPTION.
        luc_data : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        palette_v_without_u = palette_v.remove(state)
        
        # GART
        V = generalized_allocation_rejection_test(P_v__u_Y, palette_v.get_list_of_values())
    
        id_pivot = V != state.value
        V_pivot = V[id_pivot]
        J_pivot = J[id_pivot]
        
        areas = {}
        eccentricities = {}
        eccentricity_mean = {}
        eccentricity_std = {}
        
        for state_v, patch in self.patches.items():
            j = V_pivot == state_v.value
            if j.size > 0:
                areas[state_v], eccentricities[state_v] = patch.target_sample(j.sum())
                
                eccentricity_mean[state_v] = np.mean(eccentricities[state_v])
                eccentricity_std[state_v] = np.std(eccentricities[state_v])
        
        P_v__u_Y_maps = {}
        for id_v, state_v in enumerate(palette_v):
            P_v__u_Y_maps[state_v] = P_v__u_Y[:, id_v]
        
        J_used = []
        
        n_allocated = {state_v : 0 for state_v in palette_v_without_u}
        n_ghost = {state_v : 0 for state_v in palette_v_without_u}
        
        # allocation
        for id_j in np.random.choice(J_pivot.size, J_pivot.size, replace=False):
            v = V_pivot[id_j]
            state_v = palette_v._get_by_value(v)
            allocated_area, J_used_last = _weighted_neighbors(map_i_data = luc_origin_data,
                                                            map_f_data = luc_data,
                                                            map_P_vf__vi_z = P_v__u_Y_maps[state_v],
                                                            j_kernel = J_pivot[id_j],
                                                            vi = state.value,
                                                            vf = v,
                                                            patch_S = areas[state_v][id_j],
                                                            eccentricity_mean = eccentricity_mean[state_v],
                                                            eccentricity_std = eccentricity_std[state_v],
                                                            neighbors_structure = self.patches[state_v].neighbors_structure,
                                                            avoid_aggregation = self.patches[state_v].avoid_aggregation,
                                                            nb_of_neighbors_to_fill = self.patches[state_v].nb_of_neighbors_to_fill,
                                                            proceed_even_if_no_probability = self.patches[state_v].proceed_even_if_no_probability)
            
            J_used += J_used_last
            
            n_allocated[state_v] += allocated_area
            
            # if the allocation has been aborted
            if allocated_area == 0:
                n_ghost[state_v] += 1
                
            
        return(J_used, n_allocated, n_ghost)

def _compute_distance(state, data, distances_to_states):
    v_matrix = (data == state.value).astype(int)
    distances_to_states[state] = ndimage.distance_transform_edt(1 - v_matrix)