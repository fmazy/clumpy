#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:37:01 2020

@author: frem
"""

from sklearn.cluster import MiniBatchKMeans as MiniBatchKMeansSklearn

from ._cluster import _Cluster


class MiniBatchKMeans(_Cluster):
    """
    Mini-Batch K-Means clustering.
    Read more in the :ref:`User Guide <mini_batch_kmeans>`.
    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.
    init : {'k-means++', 'random'} or ndarray of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.
    max_iter : int, default=100
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.
    batch_size : int, default=100
        Size of the mini batches.
    verbose : int, default=0
        Verbosity mode.
    compute_labels : bool, default=True
        Compute label assignment and inertia for the complete dataset
        once the minibatch optimization has converged in fit.
    random_state : int, RandomState instance, default=None
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    tol : float, default=0.0
        Control early stopping based on the relative center changes as
        measured by a smoothed, variance-normalized of the mean center
        squared position changes. This early stopping heuristics is
        closer to the one used for the batch variant of the algorithms
        but induces a slight computational and memory overhead over the
        inertia heuristic.
        To disable convergence detection based on normalized center
        change, set tol to 0.0 (default).
    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini
        batches that does not yield an improvement on the smoothed inertia.
        To disable convergence detection based on inertia, set
        max_no_improvement to None.
    init_size : int, default=None
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accuracy): the
        only algorithm is initialized by running a batch KMeans on a
        random subset of the data. This needs to be larger than n_clusters.
        If `None`, `init_size= 3 * batch_size`.
    n_init : int, default=3
        Number of random initializations that are tried.
        In contrast to KMeans, the algorithm is only run once, using the
        best of the ``n_init`` initializations as measured by inertia.
    reassignment_ratio : float, default=0.01
        Control the fraction of the maximum number of counts for a
        center to be reassigned. A higher value means that low count
        centers are more easily reassigned, which means that the
        model will take longer to converge, but should converge in a
        better clustering.
    """
    
    def __init__(self, n_clusters=8, init='k-means++', max_iter=100, batch_size=100, verbose=0, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        
    def _new_estimator(self):
        return(MiniBatchKMeansSklearn(n_clusters=self.n_clusters,
                                      init=self.init,
                                      max_iter=self.max_iter,
                                      batch_size=self.batch_size,
                                      verbose=self.verbose,
                                      random_state=self.random_state))