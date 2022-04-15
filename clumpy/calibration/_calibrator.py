# -*- coding: utf-8 -*-
from time import time
from copy import deepcopy

from .._base import FeatureLayer
from .._base import State

class Calibrator():
    def __init__(self, 
                 tpe,
                 features,
                 verbose = 0):
        self.tpe = tpe
        self.features = features
        self.verbose = verbose
        
    def __repr__(self):
        return ('Calibrator(tpe='+str(self.tpe)+')')
    
    def copy(self):
        return(Calibrator(tpe=deepcopy(self.tpe),
                          features=self.features.copy(),
                          verbose=self.verbose))
    
    def fit(self,
            J,
            V,
            state,
            lul,
            distances_to_states={}):
        """
        lul est utilisé pour le calcul des distances dans features.get()
        """
        self._time_fit = {}
        
        # TIME
        # st = time()
        # # GET VALUES
        # J_calibration, X, V = land.get_values(kind='calibration',
        #                                       explanatory_variables=True,
        #                                       distances_to_states=distances_to_states)
        # self._time_fit['get_values'] = time()-st
        st = time()
                        
        if self.verbose > 0:
            print('feature selecting...')
        
        # get X
        X = self.features.fit(J=J, 
                              V=V,
                              state=state,
                              lul=lul, 
                              distances_to_states=distances_to_states,
                              return_X=True)

        if self.verbose > 0:
            print('feature selecting done.')

        self._time_fit['feature_selector'] = time()-st
        st=time()
        
        # BOUNDARIES PARAMETERS
        bounds = self.features.get_bounds()
        
        self._time_fit['boundaries_parameters_init'] = time()-st
        
        # TRANSITION PROBABILITY ESTIMATOR
        st = time()
        self.tpe.fit(X=X,
                     V=V,
                     v_initial = int(state),
                     bounds = bounds)
        self._time_fit['tpe_fit'] = time()-st
        
        return(self)
    
    def transition_probabilities(self,
                                 land,
                                 distances_to_states={},
                                 path_prefix=None,
                                 copy_geo=None,
                                 save_P_Y__v=False,
                                 save_P_Y=False,
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

        self._time_tp = {}
        st = time()
        
        state = land.state
        tm = land.region.get_transition_matrix().extract(state.value)
        
        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level) + 'Land ' + str(state) + ' TPE\n')

        # GET VALUES
        st = time()
        J, Y = land.get_values(kind='allocation',
                               explanatory_variables=True,
                               distances_to_states=distances_to_states)
        
        # features SELECTOR
        Y = self._features.selector.transform(X=Y)

        # TRANSITION PROBABILITY ESTIMATION
        P_v__u_Y = self.tpe.transition_probability(
            transition_matrix=tm,
            Y=Y,
            compute_P_Y__v=True,
            compute_P_Y=True,
            save_P_Y__v=save_P_Y__v,
            save_P_Y=save_P_Y)

        if self.verbose > 0:
            print('Land ' + str(state) + ' TPE done.\n')

        if path_prefix is not None:

            folder_path, file_prefix = path_split(path_prefix, prefix=True)

            for id_state, state_v in enumerate(tm.palette_v):
                if isinstance(lul, LandUseLayer):
                    shape = lul.get_data().shape
                    copy_geo = lul
                else:
                    shape = lul.shape
                M = np.zeros(shape)
                M.flat[J] = P_v__u_Y[:, id_state]

                file_name = file_prefix + '_' + str(state_v.value) + '.tif'

                FeatureLayer(label=file_name,
                             data=M,
                             copy_geo=copy_geo,
                             path=folder_path + '/' + file_name)

        if return_Y:
            return J, P_v__u_Y, Y
        else:
            return J, P_v__u_Y