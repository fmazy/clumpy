import numpy as np
from tqdm import tqdm
# import sparse
# from ghalton import Halton
# import sobol
import pandas as pd
from multiprocessing import Pool



from ._density_estimator import DensityEstimator
from . import bandwidth_selection

from ..utils._hyperplane import Hyperplane

# Import module
# import numpy.ctypeslib as npct
# import ctypes as ct
# mymodule = npct.load_library('libashfunc', '/home/frem/anaconda3/lib/')
# mymodule.restype = ct.c_double

import numpy as N
from numpy.ctypeslib import load_library
from . import numpyctypes

mylib = load_library('libashfunc', '/home/frem/anaconda3/lib/')       # '.' is the directory of the C++ lib

def myfunc(array1, array2):
    arg1 = numpyctypes.c_ndarray(array1, dtype=N.double, ndim = 3, shape = (4,3,2))
    arg2 = numpyctypes.c_ndarray(array2, dtype=N.double, ndim = 3, shape = (4,3,2))
    return mylib.myfunc(arg1, arg2)

# def call_multiply(arr_in, factor):
#     ''' Convenience function for converting the arguments from numpy
#         array pointers to ctypes arrays. '''
#
#     # Allocate the output array in memory, and get the shape of the array
#     arr_out = np.zeros_like(arr_in)
#     shape = np.array(arr_in.shape, dtype=np.uint32)
#
#     c_intp = ct.POINTER(ct.c_int)  # ctypes integer pointer
#     c_doublep = ct.POINTER(ct.c_double)
#     c_uintp = ct.POINTER(ct.c_uint)  # ctypes unsigned integer pointer
#
#     # Call function
#     mymodule.multiply(arr_in.ctypes.data_as(c_doublep),  # Cast numpy array to ctypes integer pointer
#                       ct.c_float(factor),  # Python integers do not need to be cast
#                       arr_out.ctypes.data_as(c_doublep),
#                       shape.ctypes.data_as(c_uintp))
#
#     return arr_out


class Digitizer():
    def __init__(self, dx, shift=0):
        self.dx = dx
        self.shift = shift

    def fit(self, X):
        self._d = X.shape[1]
        self._bins = [np.arange(V.min() - self.dx + self.shift,
                                V.max() + self.dx + self.shift,
                                self.dx) for V in X.T]

        return (self)

    def transform(self, X):
        X = X.copy()
        for k in range(self._d):
            X[:, k] = np.digitize(X[:, k], bins=self._bins[k])
        return (X.astype(int))

    def fit_transform(self, X):
        self.fit(X)

        return (self.transform(X))

    def inverse_transform(self, X_digitized):
        X = np.zeros(X_digitized.shape)
        for k in range(self._d):
            X[:,k] = self._bins[k][0] + (X_digitized[:,k] - 0.5) * self.dx
        return(X)

def box(X, h):
    dist = np.linalg.norm(X, axis=1, ord=np.inf)
    s = np.zeros(X.shape[0])
    s[dist <= h] = 1
    s *= 1 / 2 ** X.shape[1]
    return(s)

class ASH(DensityEstimator):
    def __init__(self,
                 h='scott',
                 q=10,
                 n_mc = 10000,
                 mc_seed = None,
                 bounds = [],
                 preprocessing='whitening',
                 forbid_null_value=False,
                 n_jobs = 1,
                 verbose=0,
                 verbose_heading_level=1):

        super().__init__(bounds = bounds,
                         forbid_null_value=forbid_null_value,
                         verbose=verbose,
                         verbose_heading_level=verbose_heading_level)

        self.preprocessing = preprocessing
        self.h = h
        self._h = None
        self.q = q
        self.n_mc = n_mc
        self.mc_seed = mc_seed
        self.n_jobs = n_jobs

    def __repr__(self):
        if self._h is None:
            return('ASH(h='+str(self.h)+')')
        else:
            return('ASH(h='+str(self._h)+')')

    def fit(self, X):
        # preprocessing
        self._set_data(X)

        # BOUNDARIES INFORMATIONS
        self._set_boundaries()

        # BANDWIDTH SELECTION
        if type(self.h) is int or type(self.h) is float:
            self._h = float(self.h)

        elif type(self.h) is str:
            if self.h == 'scott' or self.h == 'silverman':
                # the scott rule is based on gaussian kernel
                # the support of the gaussian kernel to have 99%
                # of the density is 2.576
                self._h = 2.576 * bandwidth_selection.scotts_rule(X)
                # self._h = bandwidth_selection.scotts_rule(X)
            else:
                raise (ValueError("Unexpected bandwidth selection method."))
        else:
            raise (TypeError("Unexpected bandwidth type."))

        if self.verbose > 0:
            print('Bandwidth selection done : h=' + str(self._h))

        # NORMALIZATION FACTOR
        self._normalization = 1 / (self._h ** self._d)

        # create a digitization for each shift
        self._digitizers = []
        self._histograms = []

        # Random Monte Carlo
        if len(self.bounds) > 0:
            np.random.seed(self.mc_seed)
            X_mc = np.random.random((self.n_mc, self._d)) * self._h - self._h / 2
            np.random.seed(None)

        self._histograms = [
            _fit_histogram(self._data, self._n, self._h, self.q, i_shift, X_mc, self._bounds_hyperplanes) for i_shift in
            tqdm(range(self.q))]


        return(self)

    def fit2(self, X):
        # preprocessing
        self._set_data(X)

        # BOUNDARIES INFORMATIONS
        self._set_boundaries()

        # BANDWIDTH SELECTION
        if type(self.h) is int or type(self.h) is float:
            self._h = float(self.h)

        elif type(self.h) is str:
            if self.h == 'scott' or self.h == 'silverman':
                # the scott rule is based on gaussian kernel
                # the support of the gaussian kernel to have 99%
                # of the density is 2.576
                self._h = 2.576 * bandwidth_selection.scotts_rule(X)
                # self._h = bandwidth_selection.scotts_rule(X)
            else:
                raise (ValueError("Unexpected bandwidth selection method."))
        else:
            raise (TypeError("Unexpected bandwidth type."))

        if self.verbose > 0:
            print('Bandwidth selection done : h=' + str(self._h))

        # NORMALIZATION FACTOR
        self._normalization = 1 / (self._h ** self._d)

        # create a digitization for each shift
        self._digitizers = []
        self._histograms = []

        # Random Monte Carlo
        if len(self.bounds) > 0:
            np.random.seed(self.mc_seed)
            X_mc = np.random.random((self.n_mc, self._d)) * self._h - self._h / 2
            np.random.seed(None)

        self._histograms = [
            _fit_histogram2(self._data, self._n, self._h, self.q, i_shift, X_mc, self._bounds_hyperplanes) for i_shift in
            tqdm(range(self.q))]


        return(self)


    def predict(self, X):
        # get indices outside bounds
        # it will be use to cut off the result later
        bounds_array = np.array(self.bounds)
        low_bound_trigger = np.array(self._low_bound_trigger)
        low_bounded_features = [int(k) for k, v  in bounds_array[low_bound_trigger]]
        low_bounds = [v for k, v in bounds_array[low_bound_trigger]]

        high_bounded_features = [int(k) for k, v in bounds_array[~low_bound_trigger]]
        high_bounds = [v for k, v in bounds_array[~low_bound_trigger]]

        id_out_of_low_bounds = np.any(X[:, low_bounded_features] < low_bounds, axis=1)
        id_out_of_high_bounds = np.any(X[:, high_bounded_features] > high_bounds, axis=1)

        if self.preprocessing != 'none':
            X = self._preprocessor.transform(X)

        f = np.zeros(X.shape[0])

        for i_shift in tqdm(range(self.q)):
            X_digitized = self._histograms[i_shift].digitizer.transform(X)

            df = pd.DataFrame(X_digitized)
            df = df.merge(self._histograms[i_shift], how='left')
            df.fillna(value=0.0, inplace=True)

            f += df.P.values

        # Normalization
        f *= self._normalization / self.q

        # outside bounds : equal to 0
        f[id_out_of_low_bounds] = 0
        f[id_out_of_high_bounds] = 0

        # Preprocessing correction
        if self.preprocessing != 'none':
            f /= np.product(self._preprocessor.scale_)

        # if null value is forbiden
        if self.forbid_null_value or self._force_forbid_null_value:
            f = self._forbid_null_values_process(f)

        return (f)

def _fit_histogram(X, n, h, q, i_shift, X_mc, bounds_hyperplanes):

    digitizer = Digitizer(dx=h,
                         shift=h / q * i_shift)
    X_digitized = digitizer.fit_transform(X)

    df = pd.DataFrame(X_digitized)
    histogram = df.groupby(by=df.columns.to_list()).size().reset_index(name='P')
    histogram['P'] /= n

    # BOUNDARIES CORRECTION
    if len(bounds_hyperplanes) > 0:
        # which cells are concerned ?
        # only close enough to the hyperplanes cells
        # are kept
        # first get cells centers
        centers = digitizer.inverse_transform(histogram[df.columns.to_list()].values)

        # then get all close enough centers
        centers_to_keep = np.zeros(centers.shape[0]).astype(bool)
        for hyp in bounds_hyperplanes:
            dist = hyp.distance(centers, p=np.inf)
            # dist = _distance_hyperplane(centers, w=bhp[0], b=bhp[1], p=np.inf)
            centers_to_keep = np.bitwise_or(centers_to_keep, dist <= h)

        I = np.array([_cell_correction(C, X_mc, bounds_hyperplanes) for C in centers[centers_to_keep]])

        # security no division by 0
        I[I == 0] = 1 / X_mc.shape[0]

        # edit the histogram with the correction
        histogram.loc[centers_to_keep, 'P'] /= I

    histogram.digitizer = digitizer
    histogram.i_shift = i_shift

    return(histogram)

def _fit_histogram2(X, n, h, q, i_shift, X_mc, bounds_hyperplanes):

    digitizer = Digitizer(dx=h,
                         shift=h / q * i_shift)
    X_digitized = digitizer.fit_transform(X)

    df = pd.DataFrame(X_digitized)
    histogram = df.groupby(by=df.columns.to_list()).size().reset_index(name='P')
    histogram['P'] /= n

    # BOUNDARIES CORRECTION
    if len(bounds_hyperplanes) > 0:
        # which cells are concerned ?
        # only close enough to the hyperplanes cells
        # are kept
        # first get cells centers
        centers = digitizer.inverse_transform(histogram[df.columns.to_list()].values)

        # then get all close enough centers
        centers_to_keep = np.zeros(centers.shape[0]).astype(bool)
        for hyp in bounds_hyperplanes:
            dist = hyp.distance(centers, p=np.inf)
            # dist = _distance_hyperplane(centers, w=bhp[0], b=bhp[1], p=np.inf)
            centers_to_keep = np.bitwise_or(centers_to_keep, dist <= h)

        I = np.array([_cell_correction(C, X_mc, bounds_hyperplanes) for C in centers[centers_to_keep]])

        # security no division by 0
        I[I == 0] = 1 / X_mc.shape[0]

        # edit the histogram with the correction
        histogram.loc[centers_to_keep, 'P'] /= I

    histogram.digitizer = digitizer
    histogram.i_shift = i_shift

    return(histogram)

def _cell_correction(C, X_mc, bounds_hyperplanes):
    n_mc = X_mc.shape[0]
    # montecarlo around the center C
    X_mc = C + X_mc
    # only elements inside the studied space are kept
    for hyp in bounds_hyperplanes:
        X_mc = X_mc[hyp.side(X_mc)]
    # the correction is equal to the ratio of kept elements
    return(X_mc.shape[0] / n_mc)
