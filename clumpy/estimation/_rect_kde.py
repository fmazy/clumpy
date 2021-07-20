#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 09:24:01 2021

@author: frem
"""

import numpy as np
from sklearn.neighbors import KDTree, BallTree
from sklearn.neighbors import NearestNeighbors
from scipy.special import gamma, betainc
from time import time
import pandas as pd
from matplotlib import pyplot as plt
import os
import itertools
from multiprocessing import Pool


from ._whitening_transformer import _WhiteningTransformer
from ..tools import _save_object, _load_object

_algorithm_class = {
    'kd_tree':KDTree,
    'ball_tree':BallTree
    }

class RectKDE():
    def __init__(self,
                 h=1.0,
                 bounded_features=[],
                 h_min = 0.1,
                 h_max = 1.0,
                 h_step = 0.01,
                 h_n_increasing = 10,
                 grid_shape = 2**8,
                 integral_tol = 0.01,
                 algorithm='kd_tree',
                 leaf_size=30,
                 n_jobs=None,
                 verbose=0):
        """
        Kernel Density Estimation (KDE) through rectangle kernel function.

        Parameters
        ----------
        h : float or {'UCV', 'UCV_mc'}, default = `1.0`
            Bandwidth.
            
            float
                set the bandwidth value
                
            'silverman'
                Silverman's rule of thumb.
            
            'UCV'
                Select the bandwidth through Unbiased Cross Validation method.
                This method is recommanded for large data set.
        
        bounded_features : list of int, default=`[]`
            List of bounded features indices. Bounded features imply some
            mirrored data which may increase computation needs. Bounded features
            are expected to be bounded by the minimum value.
        
        h_min : float, default=`0.1`
            Minimum bandwidth value. Only needed for bandwidth selection.
        
        h_max : float, default=`1.0`
            Maximum bandwidth value. Only needed for bandwidth selection.
            In case of bounded features, some mirrored data are removed
            according to this maximum bandwidth 
            (they are too far to have any impact on densities).
            
        h_step : float, default=`1.0`
            Bandwidth step value. Only needed for bandwidth selection.
            
        h_n_increasing : int, default=`10`
            Number of increasing J to break the optimal bandwidth search.
            It counts the number of steps whose
            J values after the minimum are greater than the minimum.
            If the count is greater than `h_n_increasing`, it breaks the search.
            
        grid_shape : int or list of int of shape (n_features, ), default=`2**8`
            Grid shape. If int, the same grid size is used for each feature.
            Used for grid densities and `UCV_mc` method
            (Monte-Carlo grid). The grid shape may be too large and error
            can raise to prevent computer overloading.
        
        integral_tol : float, default=`0.01`
            Integral tolerance. If integral computed over a grid is lower than
            `1 - integral_tol`, a printed warning raises.
        
        algorithm : {'kd_tree', 'ball_tree'}, default=`'kde_tree'`
            Scikit learn parameter. Nearest Neighbors algorithm.
            See Sklearn documentation.
            
        leaf_size : positive int, default=`40`
            Scikit learn parameter. Number of points at which to switch to
            brute-force algorithm.
            Changing leaf_size will not affect the result of a query, but can
            significantly impact the speed of a query and the memory required
            to store the constructed tree.
            See Sklearn documentation.
            
        dualtree : bool, default='False'
            Scikitlearn parameter. It is used within the Unbiased
            Cross-Validation bandwidth selection method to compute the
            leave one out esperance.
            If True, use a dualtree algorithm. Otherwise, use a single-tree
            algorithm. Dual tree algorithms can have better scaling for large N.
        
        verbose : int, default=`0`
            Verbosity level.

        """
        self.h = h
        self.p = 2 # no choice for this parameter
        self.bounded_features = bounded_features
        self.h_min = h_min
        self.h_max = h_max
        self.h_step = h_step
        self.h_n_increasing = h_n_increasing
        self.grid_shape = grid_shape
        self.integral_tol = integral_tol
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs=n_jobs
        self.verbose = verbose
    
    def fit(self, X):
        """
        Fit the model. Select the bandwidth according to `self.h`.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            Training data set.
            n_samples is the number of points in the data set, and n_features is
            the number of features which corresponds to the number of dimensions. 
        
        Returns
        -------
        self
        """
        if self.verbose > 0:
            print('input data shape : ', X.shape)
            
            if len(self.bounded_features) > 0:
                print('\nsymetry for bounds')
                print('bounded_features : ', self.bounded_features)
        
        #============================================================
        # 1. Data operations
        #============================================================
        
        # --------------------------------
        # 1.1. Mirror
        # --------------------------------
        
        # mirror the data is case of bounded features
        # for now, it is a full symetry
        self._first_mirror_id = X.shape[0]
        X, self._low_bounds = _mirror(X, bounded_features=self.bounded_features)
        
        if self.verbose > 0 and len(self.bounded_features) > 0:
            print('mirrored data shape : ', X.shape)
        
        # --------------------------------
        # 1.2. Support
        # --------------------------------
        
        # the support is used for grid creating
        # (notably in 'UCV_mc' h selection method)
        self._support_min = X.min(axis=0) - X.std(axis=0)
        self._support_max = X.max(axis=0) + X.std(axis=0)
        self._support_min[self.bounded_features] = self._low_bounds - X[:, self.bounded_features].std(axis=0)/2
        
        # --------------------------------
        # 1.3. Whitening transformation
        # --------------------------------
        
        # whitening transformation
        self._whitening_transformer = _WhiteningTransformer()
        # X is ereased
        X = self._whitening_transformer.fit_transform(X)
        
        # n and d are set here
        # n is not equal to self._data.shape[0] is case of mirror
        # due to mirror data selection
        self._n = X.shape[0]
        self._d = X.shape[1]
        
        # volume to apply to KDE according to the whitening transformation
        self._v = volume_unit_ball(self._d, self.p) * self._whitening_transformer._inverse_transform_det
        
        # --------------------------------
        # 1.4. Mirror data selection
        # --------------------------------
        
        if len(self.bounded_features) > 0 and self.h == 'UCV':
            # get hmax if h is given
            
            # if no bandwidth selection, the h radius is enought
            radius_ms = self.h
                
            if self.h == 'UCV':
                # if bandwidth selection, 2 * hmax radius are required
                # for exact integral squared
                radius_ms = self.h_max * 2
            
            X = _mirror_data_selection(X=X,
                                        algorithm=self.algorithm,
                                        leaf_size=self.leaf_size,
                                        first_mirror_id=self._first_mirror_id,
                                        radius=radius_ms,
                                        n_jobs=self.n_jobs)
            
            if self.verbose > 0:
                print('selected mirrored data shape : ', X.shape)
        
        # --------------------------------
        # 1.5. Data conclusion
        # --------------------------------
        # all data operations are made
        self._data = X
        
        #============================================================
        # 2. First Nearest Neighbors setting
        #============================================================
        
        # The KDE nearest neighbors tree is set a first time
        if self.verbose > 0:
            print('\nNearest neighbors tree training...')
        # Nearest Neighbors tree
        if self.algorithm not in _algorithm_class.keys():
            raise(ValueError("Unexpected algorithm parameter '"+str(self.algorithm)+"'. It should belong to {'kd_tree', 'ball_tree'}."))
        # self._tree = _algorithm_class[self.algorithm](self._data,
                                                       # leaf_size=self.leaf_size)
        
        
        
        self._nn = NearestNeighbors(algorithm=self.algorithm,
                                    leaf_size=self.leaf_size,
                                    n_jobs=self.n_jobs)
        
        if self.h == 'UCV':
            self._nn.fit(self._data)
                
        if self.verbose > 0:
            print('...done')
        
        #============================================================
        # 3. Bandwidth selection
        #============================================================
        
        if type(self.h) is str:
            if self.h == 'UCV':
                self._h = self._compute_h_through_ucv()
            elif self.h == 'silverman':
                self._h = (4 / (self._d + 2))**(1/(self._d + 4)) * self._n **(-1/(self._d + 4)) * 2
            elif self.h == 'scott':
                self._h = self._n ** (-1 / (self._d + 4)) * 2
            else:
                raise(TypeError("Unexpected h parameter type. Should be a float or {'UCV', 'silverman', 'scott'}."))
        elif type(self.h) is float:
            self._h = self.h
        else:
            raise(TypeError("Unexpected h parameter type. Should be a float or {'UCV', 'UCV_mc'}."))
                
        #============================================================
        # 4. Final training
        #============================================================
        # If their are mirrored data and the bandwidth has been selected,
        # training data should be adujsted to the new bandwidth
        # then, radius neighbors are searched for mirrored data
        if len(self.bounded_features) > 0 and self.h == 'UCV':
            self._data = _mirror_data_selection(X=self._data,
                                                algorithm=self.algorithm,
                                                leaf_size=self.leaf_size,
                                                first_mirror_id=self._first_mirror_id,
                                                radius=self._h,
                                                n_jobs=self.n_jobs)
            
            # The KDE nearest neighbors tree is set a first time
            if self.verbose > 0:
                print('\nFinal nearest neighbors tree training...')
            # Nearest Neighbors tree
            # self._tree = _algorithm_class[self.algorithm](self._data,
                                                          # leaf_size=self.leaf_size)
            self._nn.fit(self._data)
            
            if self.verbose > 0:
                print('...done')
        
        return(self)
    
    def density(self, X, h=None):
        """
        Estimate the density of a data set according to the training data set.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The data set to estimate the density.
        
        h : float, default=`None`
            The KDE bandwidth. If `None`, it takes the selected bandwidth during
            the fit method (`self._h`).

        Returns
        -------
        density : numpy array of shape (n_samples,)
            The array of density evaluations.
        """
        if not hasattr(self, '_data'):
            raise(ValueError('The kernel density estimator is not trained yet.\
                             Please call the fit method before any density \
                             request.'))
        
        if h is None:
            h = self._h
        
        # get out_of_bounds indices
        out_of_bounds_ind = np.any(X[:, self.bounded_features] < self._low_bounds, axis=1)
        
        # transform X to whitening space
        X = self._whitening_transformer.transform(X)
        
        # if self.n_jobs is None:
        #     n_jobs = 1
        # else:
        #     n_jobs = self.n_jobs
        
        # count neighbors
        density = self._nn.radius_neighbors(X=X,
                                            radius=h,
                                            return_distance=False)
        density = np.array([d.size for d in density])
        # if n_jobs == 1:
        #     density = self._tree.query_radius(X=X,
        #                                       r=h,
        #                                       count_only=True)
        # else:
        #     pool = Pool(n_jobs)
        #     density = pool.imap(self._tree.query_radius, )
        #     density = np.array(density)
        
        # divide for integral closure
        density = density / ( self._n * self._v * h**self._d)
        
        # scale in case of mirrors
        density[out_of_bounds_ind] = 0
        density = density * 2 ** len(self.bounded_features)
        
        return(density)
    
    def nearest_grid_density(self, X, h=None, dx=None):
        print(X.shape)
        min_X = np.min(X, axis=0)
        
        xs = np.round((X - min_X) / dx) * dx + min_X
        xs = np.unique(xs, axis=0)
        
        print(xs.shape)
        
        density_xs = self.density(xs, h=h) 
        
        
        
        # z = np.vstack([X_sides[0], X_sides[1][~np.isin(X_sides[1],X_sides[0])]])
        
        return(1)
    
    def grid_density(self, h=None, grid_shape=None):
        """
        Estimate the density through a grid.

        Parameters
        ----------
        h : float, default=`None`
            The KDE bandwidth. If `None`, it takes the selected bandwidth during
            the fit method (`self._h`).
            
        grid_shape : int or list of int of shape (n_features, ), default=`2**8`
            Grid shape. If int, the same grid size is used for each feature.
            Used for grid densities and `UCV_mc` method
            (Monte-Carlo grid). The grid shape may be too large and error
            can raise to prevent computer overloading. If `None`, it takes the
            constructed grid_shape value.

        Returns
        -------
        X_grid : numpy array of shape (n_grid_samples, n_features)
            Grid data set.
        
        density : numpy array of shape (n_grid_samples,)
            The array of density evaluations according to `X_grid`.
        """
        if h is None:
            h = self._h
        
        if grid_shape is None:
            grid_shape = self.grid_shape
        
        # create a grid
        X_grid = self._create_grid(grid_shape = grid_shape)
        
        # get grid density
        density = self.density(X_grid, h=h)
        
        # compute integral
        integral = density.sum() * np.product(X_grid.max(axis=0)-X_grid.min(axis=0)) / X_grid.shape[0]
        
        if self.verbose > 0:
            print('integral=',integral)
        
        # check integral validity
        _check_integral_close_to_1(integral, eps=self.integral_tol)
        
        return(X_grid, density)
    
    def _create_grid(self, grid_shape=None):
        """
        create a grid. The grid size is checked in order to not to be too large.

        """
        if grid_shape is None:
            grid_shape = self.grid_shape
        
        # if grid_shape is int, use this value for all dimensions
        if type(grid_shape) is int:
            grid_shape = (np.ones(self._d) * grid_shape).astype(int)
        
        if np.product(grid_shape) > 10*10**6:
            raise(ValueError("The grid shape is too large !\
                             Decrease some dimension size or use another\
                             un-montecarlo method for bandwidth selection such as 'UCV'"))
        
        # create linear mesh grid
        xk = np.meshgrid(*(np.linspace(self._support_min[k],self._support_max[k], grid_shape[k]) for k in range(self._d)))
        X_grid = np.vstack([xki.flat for xki in xk]).T
        
        return(X_grid)
    
    def _compute_h_through_ucv(self, real_scale=False):
        # get neighbors for J construction
        # use hmax
        st = time()
        X_real = self._data[:self._first_mirror_id]
        
        # integral squared
        if self.verbose > 0:
            print('computing distances to 2 * h_max for real data of shape '+str(X_real.shape)+'...')
        distances_2hmax, _ = self._nn.radius_neighbors(X = X_real,
                                                       radius = 2 * self.h_max,
                                                       return_distance = True)
        if self.verbose > 0:
            print('...done')
        
        self._opt_J = []
        self._opt_h = []
        
        if self.verbose > 0:
            print('starting while loop...')
        
        n_increasing = 0
        h = self.h_max + self.h_step
        while n_increasing < self.h_n_increasing and h >= self.h_min:
            h -= self.h_step
            
            # integral squared
            distances = [d[d <= 2 * h] for d in distances_2hmax]
            # return(distances)
            
            # hyper sphere intersection volume
            # see https://math.stackexchange.com/questions/162250/how-to-compute-the-volume-of-intersection-between-two-hyperspheres
            
            hsiv = np.sum([np.sum(1 - betainc(1/2, (self._d+1)/2, delta**2 / 4 / h**2)) for delta in distances])
            # warning ! h**d normaly !
            hsiv *= np.pi**(self._d/2) * h**self._d / gamma(self._d/2+1)
            hsiv *= self._whitening_transformer._inverse_transform_det                                                             
            # scale returned volume for mirrors considerations
            hsiv *= 2**len(self.bounded_features)
            
            # integral closure
            integral_squared = 1 / (self._n * h**(2 * self._d) * self._v) * hsiv
            
            if real_scale:
                # if real scale, some divisions are required
                integral_squared /= self._n * self._v
                
            # leave one out esperance
            s = np.sum([(d <= h).sum() for d in distances_2hmax])
            # one removes auto-paired points since the data and the training
            # set are the same
            s -= self._first_mirror_id
            
            # mirror considerations
            s *= 2**len(self.bounded_features)
            
            # integral closure
            s = s / ((self._n - 1) * h**self._d)
            
            if real_scale:
                # if real scale, some divisions are required
                s /= self._n * self._v
            
            J = integral_squared - 2 * s
            
            self._opt_J.append(J)
            self._opt_h.append(h)
            
            min_ind = np.argmin(self._opt_J)
            min_value = np.min(self._opt_J)
            n_increasing = np.sum(np.array(self._opt_J)[min_ind:]>min_value)
            
            if self.verbose > 0:
                print(self._opt_h[-1], self._opt_J[-1], n_increasing)
        
        self._opt_time = time()-st
        
        return(self._opt_h[np.argmin(self._opt_J)])
    
    # def _compute_h_through_ucv2(self, montecarlo=False, real_scale=False):
    #     """
    #     UCV for loop method.
    #     compute J for several bandwidth and return the optimal bandwidth
    #     which minimize J.
    #     """
    #     if self.verbose > 0:
    #         print("\nComputing optimal h through UCV")
    #         print("montecarlo method : ", montecarlo)
    #         print("real scale : ", real_scale)
    #         print("h min : ", self.h_min)
    #         print("h max : ", self.h_max)
    #         print("h step : ", self.h_step)
    #         print("num. of increasing J to break the research : ", self.h_n_increasing)
    #         print("\nstart\n")
    #         print("h | J | n_increasing")
    #     if montecarlo:
    #         # if montecarlo, create a grid.
    #         # the grid is thus created only one time.
    #         X_grid = self._create_grid()
    #     else:
    #         X_grid = None
        
    #     # linear bandwidth variation 
    #     self._opt_h = np.arange(start = self.h_min,
    #                        stop = self.h_max,
    #                        step = self.h_step)
    #     self._opt_J = []
        
    #     # n increasing counting initialization
    #     n_increasing = 0
        
    #     # start time for execution time
    #     st = time()
    #     for h in self._opt_h:
    #         # compute J and append it to _opt_J
    #         # the real scale is set to False. The real value of J is not
    #         # required. Comparisons are enought
    #         self._opt_J.append(self._compute_J(h, X_grid, real_scale=real_scale))
            
    #         # n increasing counting. it counts the number of steps whose
    #         # J values after the minimum are greater than the minimum
    #         min_ind = np.argmin(self._opt_J)
    #         min_value = np.min(self._opt_J)
    #         n_increasing = np.sum(np.array(self._opt_J)[min_ind:]>min_value)
            
    #         if self.verbose>0:
    #             print(h, self._opt_J[-1], n_increasing)
            
    #         # if the number of consecutively increasing J is triggered
    #         # break the for loop.
    #         # two possibilities : 1/ the hmin was to high
    #         # 2/ the optimal h has been observed and is the argmin of J
    #         if n_increasing >= self.h_n_increasing:
    #             if self.verbose > 0:
    #                 print('num. of J increasing reached. Break the research.')
    #             break
        
    #     self._opt_h = self._opt_h[:len(self._opt_J)]
        
    #     # execution time
    #     self._opt_time = time() - st
        
    #     # opt_J as a numpy array
    #     self._opt_J = np.array(self._opt_J)
        
    #     if self.verbose > 0:
    #         print('Optimal h computing through UCV done.\n')
        
    #     return(self._opt_h[np.argmin(self._opt_J)])
        
    # def _compute_J(self, h, X_grid=None, real_scale=True):
    #     """
    #     UCV method. Compute an estimation of J 
    #     """
    #     # compute integral squared. If montecarlo, X_grid is not None
    #     integral_squared = self._compute_integral_squared(h, X_grid, real_scale=real_scale)
        
    #     # compute leave one out esperance
    #     leave_one_out_esperance = self._compute_leave_one_out_esperance(h, real_scale=real_scale)
        
    #     # compute J
    #     J = integral_squared - 2 * leave_one_out_esperance
        
        
    #     return(J)
    
    # def _compute_integral_squared(self, h, X_grid=None, real_scale=True):
    #     """
    #     compute integral squared. switch to 2 different methods :
    #     with montecarlo approximation or not.
    #     """
    #     # montecarlo switch
    #     if X_grid is None:
    #         return(self._compute_exact_integral_squared(h, real_scale=real_scale))
    #     else:
    #         return(self._compute_mc_integral_squared(h, X_grid, real_scale=real_scale))
    
    # def _compute_mc_integral_squared(self, h, X_grid, real_scale=True):
    #     """
    #     compute integral squared through monte carlo approximation
    #     """
    #     # montecarlo coefficient according to the original space
    #     mc_coef = np.product(X_grid.max(axis=0)-X_grid.min(axis=0)) / X_grid.shape[0]
                
    #     # count neighbors through the density function with set h
    #     p_grid = self.density(X_grid, h=h)
        
    #     # compute integral
    #     integral = p_grid.sum() * mc_coef
        
    #     # integral check
    #     # a warning message can be displayed but it is not considered as
    #     # an error
    #     _check_integral_close_to_1(integral, eps=self.integral_tol)
        
    #     # compute integral squared
    #     integral_squared = (p_grid**2).sum() * mc_coef
        
    #     # mirrors consideration
    #     # the p_grid has already been scaled for mirror considerations
    #     # It have to been then downscaled according to formulas.
    #     # indeed, the integral is made on the whole support
    #     # without mirror considerations.
    #     integral_squared /= 2**len(self.bounded_features)
        
        
    #     if not real_scale:
    #         # in case of no real scale, some scaling are required
    #         integral_squared *= self._n * self._v
        
    #     return(integral_squared)
    
    # def _compute_exact_integral_squared(self, h, real_scale=True):
    #     """
    #     compute integral squared through exact formulas.
    #     """
        
    #     # get neighbors distances
    #     indices, distances = self._tree.query_radius(X = self._data[:self._first_mirror_id],
    #                                                   r = 2 * h,
    #                                                   return_distance=True)
        
    #     # compute hypersphere intersection volume
    #     # all pairs of points are taken 2 times. it is in accordance
    #     # with the integral squared formula
    #     hypersphere_intersection_volume = hyperspheres_inter_volume(distances=distances,
    #                                                                 radius=h,
    #                                                                 n_dims=self._d) * self._whitening_transformer._inverse_transform_det
    #     # scale returned volume for mirrors considerations
    #     hypersphere_intersection_volume *= 2**len(self.bounded_features)
        
    #     # integral closure
    #     integral_squared = 1 / (self._n * h**(2 * self._d) * self._v) * hypersphere_intersection_volume
        
    #     if real_scale:
    #         # if real scale, some divisions are required
    #         integral_squared /= self._n * self._v
        
    #     return(integral_squared)
        
    # def _compute_leave_one_out_esperance(self, h, real_scale=True):
    #     """
    #     compute leave one out esperance which is used to estimate J.
    #     """
    #     # count pairs of points
    #     s = self._tree.two_point_correlation(X=self._data[:self._first_mirror_id],
    #                                          r=h,
    #                                          dualtree=self.dualtree)[0]
    #     # one should remove auto-paired points since the data and the training
    #     # set are the same
    #     s -= self._first_mirror_id
        
    #     # mirror considerations
    #     s *= 2**len(self.bounded_features)
        
    #     # integral closure
    #     s = s / ((self._n - 1) * h**self._d)
        
    #     if real_scale:
    #         # if real scale, some divisions are required
    #         s /= self._n * self._v
        
    #     return(s)

    def plot_h_opt(self):
        """
        Plot the bandwidth selection process.

        Returns
        -------
        plt : matplotlib.pyplot object

        """
        df = pd.DataFrame(self._opt_h, columns=['h'])
        df['J'] = self._opt_J
        df.sort_values(by='h', inplace=True)
        
        plt.plot(df.h, df.J, label='opt algo')
        plt.vlines(self._h, ymin=df.J.min(), ymax=df.J.max(), color='red', label='selected value')
        plt.xlabel('h')
        plt.ylabel('$\hat{J}$')
        plt.legend()
        
        return(plt)

    def save(self, path):
        """
        Save the RectKDE object.

        Parameters
        ----------
        path : str
            File path. A 'zip' file is expected.
            Directories are created if needed.

        Returns
        -------
        success : bool
            `True` is the saving process is a success.
        """
        folder_name = os.path.dirname(path)
        files_names = []
                
        files_names.append('kde.zip')
        _save_object(self, 'kde.zip')
        
        files_names.append('whitening_transformer.zip')
        _save_object(self._whitening_transformer, 'whitening_transformer.zip')
        
        # create output directory if needed
        if folder_name != "":
            os.system('mkdir -p ' + folder_name)
        
        # zip file
        command = 'zip '+path
        for file_name in files_names:
            command += ' ' + file_name
        os.system(command)
        
        # remove files        
        command = 'rm '
        for file_name in files_names:
            command += ' ' + file_name
        os.system(command)
        
        return(self)
    
    def load(self, path):
        """
        Load a RectKDE object.

        Parameters
        ----------
        path : str
            File path. A 'zip' file is expected with two other 'zip' file
            encapsuled inside :
                'kde.zip'
                'whitening_transformer.zip'

        Returns
        -------
        success : bool
            `True` is the loading process is a success.
        """
        os.system('unzip ' + path + ' -d ' + path + '.kde_out')
        
        files = os.listdir(path + '.kde_out/')
        
        for file in files:
            if file == 'kde.zip':
                _load_object(self, path+'.kde_out/kde.zip')
            
            elif file == 'whitening_transformer.zip':
                self._whitening_transformer = _WhiteningTransformer()
                _load_object(self._whitening_transformer, path+'.kde_out/whitening_transformer.zip')
                
        os.system('rm -R ' + path + '.kde_out')
        
        self._nn = NearestNeighbors(algorithm=self.algorithm,
                                    leaf_size=self.leaf_size,
                                    n_jobs=self.n_jobs)
        self._nn.fit(self._data)
        
        # if not hasattr(self, '_h') and hasattr(self, '_opt_h') and hasattr(self, '_opt_J'):
        #     self._h = self._opt_h[np.argmin(self._opt_J)]
        
        # self._tree = _algorithm_class[self.algorithm](self._data,
                                                      # leaf_size=self.leaf_size)
        
        # if not hasattr(self._whitening_transformer, '_transform_det'):
        #     self._whitening_transformer._transform_det = np.abs(np.linalg.det(self._whitening_transformer._transform_matrix))
        # self._whitening_transformer._inverse_transform_det = np.abs(np.linalg.det(self._whitening_transformer._inverse_transform_matrix))
        
        # if not hasattr(self, '_v') and hasattr(self, '_n') and hasattr(self, '_d'):
        #     self._v = volume_unit_ball(self._d, self.p) * self._whitening_transformer._inverse_transform_det
        
        return(self)

def _mirror(X, bounded_features):
    """
    mirror the data set according to bounded features.
    """
    low_bounds = X[:, bounded_features].min(axis=0)
    
    for idx, feature in enumerate(bounded_features):
        X_mirrored = X.copy()
        X_mirrored[:, feature] = 2 * low_bounds[idx] - X[:, feature]

        X = np.vstack((X, X_mirrored))
    
    return(X, low_bounds)

def _mirror_data_selection(X, algorithm, leaf_size, first_mirror_id, radius, n_jobs):
    """
    select mirrored data set according to rhe bandwidth.
    only close enought mirrored data are kept.
    """
    # mirror data selection
    
    # tree_ms -> nearest neighbors tree mirror selection
    # trained through original data
    # tree_ms = _algorithm_class[algorithm](X[:first_mirror_id],
                                          # leaf_size=leaf_size)
    nn = NearestNeighbors(radius = radius,
                          algorithm = algorithm,
                          leaf_size = leaf_size,
                          n_jobs = n_jobs)
    # train through real data
    nn.fit(X[:first_mirror_id])
    # count neighbors for mirrored data
    count_neighbors = nn.radius_neighbors(X=X[first_mirror_id:],
                                          radius=radius,
                                          return_distance=False)
    count_neighbors = np.array([cn.size for cn in count_neighbors])
    # count_neighbors = tree_ms.query_radius(X=X[first_mirror_id:],
                                           # r=radius,
                                           # count_only=True)
    
    # only pixels with neighbors are kept
    ind_to_keep = first_mirror_id + np.arange(len(count_neighbors))[count_neighbors > 0]
    # selected mirrored data
    X = np.vstack((X[:first_mirror_id],
                    X[ind_to_keep]))
    
    return(X)
    
def volume_unit_ball(d, p=2):
    """
    Volume of d-dimensional unit ball under the p-norm. When p=1 this is called
    a cross-polytype, when p=2 it's called a hypersphere, and when p=infty it's
    called a hypercube.

    Notes
    -----
    See the following paper for a very general result related to this:

    - Wang, Xianfu. “Volumes of Generalized Unit Balls.”
      Mathematics Magazine 78, no. 5 (2005): 390–95.
      https://doi.org/10.2307/30044198.
    """
    return 2.0 ** d * gamma(1 + 1 / p) ** d / gamma(1 + d / p)

def hyperspheres_inter_volume_part(distances, radius, n_dims):
    return(2*np.array([Vn(radius, dist/2, n_dims) for dist in distances]))

def hyperspheres_inter_volume(distances, radius, n_dims):
    """
    compute the total hypersphere intersection volume of a list of distances.
    radius are considered as fixed and equals
    see https://math.stackexchange.com/questions/162250/how-to-compute-the-volume-of-intersection-between-two-hyperspheres
    """
    return(2*np.sum([Vn(radius, dist/2, n_dims).sum() for dist in distances]))

def Vn(r, a, n):
    """
    function used in the hypersphere intersection volume.
    see https://math.stackexchange.com/questions/162250/how-to-compute-the-volume-of-intersection-between-two-hyperspheres
    """
    return(1/2*np.pi**(n/2)*r**n*betainc((n+1)/2, 1/2, 1-a**2/r**2)/gamma(n/2+1))

def _check_integral_close_to_1(integral, eps=1e-2):
    """
    Check if the integral is close enought to 1.
    It prints a warning if not.
    """
    if np.abs(1-integral) > eps:
        print("/!\ WARNING /!\ Integral="+str(integral)+"\nwhich is too far from 1.\nThe grid density should be increased.")
        return(False)
    else:
        return(True)

