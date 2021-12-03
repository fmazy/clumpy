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

        # pfiou, tenter de paralléliser est un échec.
        # peut etre que la fonction est trop grosse.
        # peut etre que Pandas est incompatible
        # peut etre ne paraléliser uniquement que le calcul de la correction de bords...
        # en tout cas il faudrait revenir à un code plus propre !

        pool = Pool(self.n_jobs)




        bounds_hyperplanes_params = [(hyp.w, hyp.b, hyp.positive_side_scalar) for hyp in self._bounds_hyperplanes]

        # self._histograms = pool.starmap_async(_fit_histogram, [(self._data,
        #                                                   self._n,
        #                                                   self._h,
        #                                                   self.q,
        #                                                   i_shift,
        #                                                   X_mc,
        #                                                   bounds_hyperplanes_params) for i_shift in range(self.q)]).get()

        args_iter = [(self._data,
                      self._n,
                      self._h,
                      self.q,
                      i_shift,
                      X_mc,
                      bounds_hyperplanes_params) for i_shift in range(self.q)]

        self._histograms = pool.map_async(
                                            _fit_histogram,
                                            iterable=args_iter,
                                            chunksize=self.n_jobs
                                        ).get()

        # self._histograms = [
        #     _fit_histogram(self._data, self._n, self._h, self.q, i_shift, X_mc, bounds_hyperplanes_params) for i_shift in
        #     range(self.q)]

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

def _fit_histogram(args):
    X, n, h, q, i_shift, X_mc, bounds_hyperplanes_params = args

    digitizer = Digitizer(dx=h,
                         shift=h / q * i_shift)
    X_digitized = digitizer.fit_transform(X)

    df = pd.DataFrame(X_digitized)
    histogram = df.groupby(by=df.columns.to_list()).size().reset_index(name='P')
    histogram['P'] /= n

    # bounds_hyperplanes = [Hyperplane(w=bhp[0],
    #                                 b=bhp[1],
    #                                 positive_side_scalar=bhp[2]) for bhp in bounds_hyperplanes_params]

    # BOUNDARIES CORRECTION
    if len(bounds_hyperplanes_params) > 0:
        # which cells are concerned ?
        # only close enough to the hyperplanes cells
        # are kept
        # first get cells centers
        centers = digitizer.inverse_transform(histogram[df.columns.to_list()].values)

        # then get all close enough centers
        centers_to_keep = np.zeros(centers.shape[0]).astype(bool)
        for bhp in bounds_hyperplanes_params:
            # dist = hyp.distance(centers, p=np.inf)
            dist = _distance_hyperplane(centers, w=bhp[0], b=bhp[1], p=np.inf)
            centers_to_keep = np.bitwise_or(centers_to_keep, dist <= h)

        I = np.array([_cell_correction(C, X_mc, bounds_hyperplanes_params) for C in centers[centers_to_keep]])

        # security no division by 0
        I[I == 0] = 1 / X_mc.shape[0]

        # edit the histogram with the correction
        histogram.loc[centers_to_keep, 'P'] /= I

    histogram.digitizer = digitizer
    histogram.i_shift = i_shift

    return(histogram)

def _distance_hyperplane(X, w, b, p):
    dist = np.abs(np.dot(X, w) + b) / np.linalg.norm(w, ord=p)

    return (dist)

def _side_hyperplane(X, w, b, positive_side_scalar):
    norm_vec = np.dot(X, w) + b
    norm_vec *= positive_side_scalar

    return(norm_vec>0)

def _cell_correction(C, X_mc, bounds_hyperplanes_params):
    n_mc = X_mc.shape[0]
    # montecarlo around the center C
    X_mc = C + X_mc
    # only elements inside the studied space are kept
    for bhp in bounds_hyperplanes_params:
        X_mc = X_mc[_side_hyperplane(X_mc, bhp[0], bhp[1], bhp[2])]
    # the correction is equal to the ratio of kept elements
    return(X_mc.shape[0] / n_mc)
