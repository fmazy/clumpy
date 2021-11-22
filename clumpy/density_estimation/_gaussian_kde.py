#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 20:16:24 2021

@author: frem
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import erf
from multiprocessing import Pool
from matplotlib import pyplot as plt

from tqdm import tqdm

from . import bandwidth_selection
from ._density_estimator import DensityEstimator

from ..utils._hyperplane import Hyperplane
from ..tools._console import title_heading

import sys

def restart_line():
    sys.stdout.write('\r')
    sys.stdout.flush()

def _gaussian(x):
    return(np.sum(np.exp(-0.5 * x**2)))

def _compute_gaussian_kde(nn, X, h, support_factor, i=None, n_steps=None, verbose=0):
    distances, neighbors_id = nn.radius_neighbors(X, radius=h * support_factor, return_distance=True)

    # ici on pourrait mettre de quoi vérifier si il y a assez de voisins
    # pour faire l'estimation.
    # avec un parametre n_min_neighbors
    #
    
    if i is not None and n_steps is not None and verbose>0:
        restart_line()
        sys.stdout.write(str(i)+'/'+str(n_steps))
        sys.stdout.flush()
                
    return(np.array([_gaussian(dist/h) for dist in distances]))

class GKDE(DensityEstimator):
    """
    Gaussian Kernel Density Estimator. It is a child of sklearn.BaseEstimator.

    Parameters
    ----------
    h : float or {'scott', 'silverman'}, default='scott'
         The bandwidth. It can be a float or a method :
             scott : The scott's rule :math:`h=-1 / n^{d+4}`.
             
             silverman : The Silverman's rule which is the same as Scott's rule.
             
    low_bounded_features : list of int, default=[]
        The low bounded features indices.
    
    high_bounded_features : list of int, default=[]
        The high bounded features indices.
    
    low_bounds : list of float, default=[]
        Consequently to ``low_bounded_features`` value, the low bounds are
        specified here.
    
    high_bounds : list of float, =[]
        Consequently to ``high_bounded_features`` value, the high bounds are
        specified here.
        
    preprocessing : None or {'whitening'}, default='whitening'
        Preprocessing transformation.
        
            whitening : Whitening transformation to get a covariance matrix equal to the identity matrix.
    
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default=``'kd_tree'``
        Algorithm used to compute the nearest neighbors as sklearn.
        
            'ball_tree' will use BallTree
            
            'kd_tree' will use KDTree
            
            'brute' will use a brute-force search.
            
            'auto' will attempt to decide the most appropriate algorithm based on the values passed to fit method.
        
        Note: fitting on sparse input will override the setting of this parameter, using brute force.
        
    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.

    support_factor : float, default=3
        The kernel support factor, even if kernel has infinite support.

    n_fit_max : int, default=2*10**4
        The maximum number of training samples. A random sampling is done if necessary.

    n_predict_max : int, default=2*10**4
        The maximum of simultaneous probability estimations.
        Several streams are consequently created.

    forbid_null_value : bool, default=False
        If ``True``, the null value is forbiden and a correction is made if
        necessary.

    n_jobs_predict : int, default=2
        The number of parallel jobs to predict according to streams defined
        by ``n_predict_max``.

    n_jobs_neighbors : int, default=1
        The number of parallel jobs to run for neighbors search.

    verbose : int, default=0
        The verbosity level.

    verbose_heading_level : int, default=1
        Verbose heading level for markdown titles. If ``0``, no markdown title are printed.

    """
    def __init__(self,
                 h='scott',
                 low_bounded_features=[],
                 high_bounded_features=[],
                 low_bounds = [],
                 high_bounds = [],
                 preprocessing='whitening',
                 algorithm='kd_tree',
                 leaf_size=30,
                 support_factor=3,
                 n_fit_max = 2*10**4,
                 n_predict_max = 2*10**4,
                 forbid_null_value = False,
                 n_jobs_predict=2,
                 n_jobs_neighbors=1,
                 verbose=0,
                 verbose_heading_level=1):
        
        super().__init__(low_bounded_features=low_bounded_features,
                         high_bounded_features=high_bounded_features,
                         low_bounds = low_bounds,
                         high_bounds = high_bounds,
                         forbid_null_value = forbid_null_value,
                         verbose=verbose,
                         verbose_heading_level=verbose_heading_level)
        
        self.h = h
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.support_factor = support_factor
        self.n_fit_max = n_fit_max
        self.n_predict_max = n_predict_max
        self.n_jobs_predict = n_jobs_predict
        self.n_jobs_neighbors = n_jobs_neighbors
        self.preprocessing = preprocessing

        self._h = None
        
    def __repr__(self):
        if self._h is None:
            return('GKDE(h='+str(self.h)+')')
        else:
            return('GKDE(h='+str(self._h)+')')

    
    def fit(self, X, y=None):
        """
        Fit according to observations

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The observations.
        
        y : None
            Not used. It is provided for compatibility only.

        Returns
        -------
        self: GKDE
            The fitted GKDE object.

        """

        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level)+'GKDE fitting...')

        if X.shape[0] > self.n_fit_max:
            X = X[np.random.choice(a=X.shape[0],
                                   size=self.n_fit_max,
                                   replace=False)]

            self._force_forbid_null_value = True

            if self.verbose > 0:
                print('n_fit_max overshoot. Random sampling done.')

        # preprocessing
        self._set_data(X)

        if self.verbose > 0:
            print('Preprocessing done.')

        # BOUNDARIES INFORMATIONS
        self._set_boundaries()
            
        if self.verbose > 0:
            print('Boundaries initialization done.')

        # BANDWIDTH SELECTION
        if type(self.h) is int or type(self.h) is float:
            self._h = float(self.h)
        
        elif type(self.h) is str:
            if self.h == 'scott' or self.h == 'silverman':
                self._h = bandwidth_selection.scotts_rule(X)
            else:
                raise(ValueError("Unexpected bandwidth selection method."))
        else:
            raise(TypeError("Unexpected bandwidth type."))

        if self.verbose > 0:
            print('Bandwidth selection done : h='+str(self._h))
        
        # NORMALIZATION FACTOR
        self._normalization = 1 / ((2*np.pi)**(self._d/2) * self._h**self._d)
        
        # NEAREST NEIGHBOR FITTING
        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level)+'sklearn.neighbors.NearestNeighbors fitting...')
        self._nn = NearestNeighbors(radius = self._h * self.support_factor,
                                   algorithm = self.algorithm,
                                   leaf_size = self.leaf_size,
                                   n_jobs = self.n_jobs_neighbors)
        self._nn.fit(self._data)

        if self.verbose > 0:
            print('sklearn.neighbors.NearestNeighbors fitting done.')
            print('GKDE fitting done.')
        return(self)
        
    def predict(self, X):
        """
        Estimate the requested probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The samples to estimate the probability.

        Returns
        -------
        f : ndarray of shape (n_samples,)
            The estimated probabilities.
        """
        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level)+'GKDE Predict')

        pdf = self._predict_with_bandwidth(X, self._h)

        if self.verbose > 0:
            print('GKDE predict done.')
        return(pdf)
    
    def _predict_with_bandwidth(self, X, h):
        """
        Private method.
        Predict with a specified bandwidth

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The samples to estimate the probability.
        h : float
            The bandwidth.

        Returns
        -------
        f : ndarray of shape (n_samples,)
            The estimated probabilities.

        """
        # get indices outside bounds
        # it will be use to cut off the result later
        id_out_of_low_bounds = np.any(X[:, self.low_bounded_features] < self.low_bounds, axis=1)
        id_out_of_high_bounds = np.any(X[:, self.high_bounded_features] > self.high_bounds, axis=1)
        
        if self.preprocessing != 'none':
            X = self._preprocessor.transform(X)

        # STEPS INITIALIZATION
        # requested elements are not estimated alltogether in order to
        # improve numerical computations.
        # the limit is given by n_predict_max
        steps = np.arange(0, X.shape[0], self.n_predict_max)

        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level)+'Gaussian kernel')
            print('process with '+str(steps.size)+' streams computed in parallel through '+str(self.n_jobs_predict)+' CPUs.')

        if self.n_jobs_predict == 1:
            if self.verbose > 0:
                steps = tqdm(steps)
            
            # for each set of n_predict_max elements
            f = np.zeros(X.shape[0])
            for i in steps:
                f[i:i+self.n_predict_max] = _compute_gaussian_kde(self._nn,
                                                                  X[i:i+self.n_predict_max],
                                                                  h,
                                                                  self.support_factor)
                    
        else:
            pool = Pool(self.n_jobs_predict)
            f = pool.starmap(_compute_gaussian_kde, [(self._nn, 
                                                      X[i:i+self.n_predict_max],
                                                      h,
                                                      self.support_factor,
                                                      id_i,
                                                      steps.size,
                                                      self.verbose) for id_i, i in enumerate(steps)])
            f = np.hstack(f)
            if self.verbose > 0:
                restart_line()
                sys.stdout.write('done\n')
                sys.stdout.flush()

        if self.verbose > 0:
            print('Gaussian kernel process done.')

        # Normalization
        f *= self._normalization / self._n
        
        # boundary bias correction
        if self.verbose > 0:
            print(title_heading(self.verbose_heading_level)+'Boundary bias correction...')

        f /= self._boundary_correction(X, h)

        # outside bounds : equal to 0
        f[id_out_of_low_bounds] = 0
        f[id_out_of_high_bounds] = 0

        if self.verbose > 0:
            print('Boundary bias correction done.')
        
        # Preprocessing correction
        if self.preprocessing != 'none':
            f /= np.product(self._preprocessor.scale_)

        # if null value is forbiden
        if self.forbid_null_value or self._force_forbid_null_value:
            if self.verbose > 0:
                print(title_heading(self.verbose_heading_level)+'Null value correction...')
            idx = f == 0.0

            m_0 = idx.sum()

            new_n = self._n + m_0

            f = f * self._n / new_n

            min_value = 1 / new_n * self._normalization * _gaussian(0)
            f[f == 0.0] = min_value

            # Warning flag
            # check the relative number of corrected probabilities
            if self.verbose > 0:
                print('m_0 = '+str(m_0)+', m = '+str(self._n)+', m_0 / m = '+str(np.round(m_0/self._n,4)))

            # warning flag
            if m_0 / self._n > 0.01:
                print('WARNING : m_0/m > 0.01. The parameter `n_fit_max` should be higher.')

            if self.verbose > 0:
                print('Null value correction done for '+str(m_0)+' elements.')

        return(f)

    def _boundary_correction(self, X, h):
        """
        X in the WT space.
        """
        correction = np.ones(X.shape[0])
        for hyperplane in self._low_bounds_hyperplanes + self._high_bounds_hyperplanes:
            correction *= 1 / 2 * (1 + erf(hyperplane.distance(X) / h / np.sqrt(2)))

        return(correction)

    # def marginal(self,
    #              x,
    #              k):
    #     X = np.random.random((x.size, self._d))
    #     X[:, k] = x
    #
    #     if self.preprocessing != 'none':
    #         X = self._preprocessor.transform(X)
    #     x = X[:, k]
    #
    #     return(x)

    def extract_marginal(self, k):
        # make list if necessary
        if not isinstance(k, list):
            k = [k]

        # if preprocessing, get original data
        if self.preprocessing != 'none':
            data = self._preprocessor.inverse_transform(self._data)

        # get only requested columns
        data = data[:,k]

        # create new marginal GKDE
        low_bounded_features = []
        high_bounded_features = []
        low_bounds = []
        high_bounds = []

        for id_marg, ki in enumerate(k):
            if ki in self.low_bounded_features:
                id_self = self.low_bounded_features.index(ki)
                low_bounded_features.append(id_marg)
                low_bounds.append(self.low_bounds[id_self])

            if ki in self.high_bounded_features:
                id_self = self.high_bounded_features.index(ki)
                high_bounded_features.append(id_marg)
                high_bounds.append(self.high_bounds[id_self])

        marg_gkde = GKDE(h=self._h,
                         low_bounded_features=low_bounded_features,
                         high_bounded_features=high_bounded_features,
                         low_bounds = low_bounds,
                         high_bounds = high_bounds,
                         preprocessing=self.preprocessing,
                         algorithm=self.algorithm,
                         leaf_size=self.leaf_size,
                         support_factor=self.support_factor,
                         n_fit_max = self.n_fit_max,
                         n_predict_max = self.n_predict_max,
                         forbid_null_value = self.forbid_null_value,
                         n_jobs_predict=self.n_jobs_predict,
                         n_jobs_neighbors=self.n_jobs_neighbors,
                         verbose=self.verbose,
                         verbose_heading_level=self.verbose_heading_level)

        marg_gkde.fit(data)

        return(marg_gkde)

    def marginal(self,
                 n,
                 k):
        """
        Estimate the marginal probability.

        Parameters
        ----------
        x : array-like of shape (n_samples,)
            The samples to estimate along the `$k$` feature.
        k : int
            The feature index.

        Returns
        -------
        f : ndarray of shape (n_samples,)
            The estimated marginal probabilities.
        """

        x_max = self._data[:,[k]].max() + self._h * self.support_factor
        x_min = self._data[:, [k]].min() - self._h * self.support_factor

        x = np.linspace(x_min, x_max, n)
        
        if self.preprocessing != 'none':
            X = self._preprocessor.transform(X)
        x = X[:, k]
        
        nn = NearestNeighbors(radius=self._h * self.support_factor,
                              algorithm = self.algorithm,
                              leaf_size=self.leaf_size,
                              n_jobs = self.n_jobs_neighbors)
        nn.fit(self._data[:,[k]])
        
        distances, _ = nn.radius_neighbors(x[:,None],
                                           radius= self._h * self.support_factor,
                                           return_distance=True)
        
        f = 1 / (2 * np.pi)**(1/2) / self._h / self._n * np.array([_gaussian(dist / self._h) for dist in distances])
        
        if k in self.low_bounded_features:
            id_k = self.low_bounded_features.index(k)
            f /= 1 / 2 * (1 + erf((x - self._low_bounds[id_k]) / self._h / np.sqrt(2)))
            
            f[x < self._low_bounds[id_k]] = 0
        
        if k in self.high_bounded_features:
            id_k = self.high_bounded_features.index(k)
            f /= 1 / 2 * (1 + erf((self._high_bounds[id_k] - x) / self._h / np.sqrt(2)))
            
            f[x > self._high_bounds[id_k]] = 0

        if self.preprocessing != 'none':
            f /= np.product(self._preprocessor.scale_)
        
        return(f)
    
    def cut_plot(self,
                 x=None,
                 q=None,
                 n_eval=300,
                 linestyle=None,
                 color=None,
                 label_prefix=''):
        """
        Estimate the value along a cut.

        """
        # check
        if (x is None and q is None) or (x is not None and q is not None):
            raise(ValueError("Either x or q are expected."))
        
        if x is not None:
            if np.sum(np.array(x) == None) != 1:
                raise(ValueError("Only one undefined features is expected. Example: x=[0.2, None, 0.1, 0.6]"))
            
            label='x='+str(x)
            
            k = x.index(None)
            k_bar = np.delete(np.arange(len(x)), k)
            
            x = self._preprocessor.transform(np.array([x]))[0]
            x[k] = None
            
            
            
        if q is not None:
            if np.sum(np.array(q) == None) != 1:
                raise(ValueError("Only one undefined features is expected. Example: x=[0.2, None, 0.1, 0.6]"))
            
            label='q='+str(q)
            
            k = q.index(None)
            k_bar = np.delete(np.arange(len(q)), k)
            
            q[k] = 0
            
            x = [np.quantile(self._data[:,i], qi) for i, qi in enumerate(q)]
            x = np.array(x)
            x[k] = None
            
            
        
        # print(k_bar)
        X_eval = np.ones((n_eval, self._d))
        X_eval[:, k_bar] *= x[k_bar]
        
        if k in self.low_bounded_features:
            a = self._low_bounds[self.low_bounded_features.index(k)]
        else:
            a = self._data[:,k].min() - self.support_factor * self._h
        
        if k in self.high_bounded_features:
            b = self._high_bounds[self.high_bounded_features.index(k)]
        else:
            b = self._data[:,k].max() + self.support_factor * self._h
        
        X_eval[:, k] = np.linspace(a, b, n_eval)
       
        X_eval = self._preprocessor.inverse_transform(X_eval)
       
        pred = self.predict(X_eval)
        
        plt.plot(X_eval[:,k], pred,
                 label=label_prefix+label,
                 color=color,
                 linestyle=linestyle)
        
        return(X_eval, pred, plt)
    


