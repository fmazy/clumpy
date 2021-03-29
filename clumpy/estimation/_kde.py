from sklearn.base import BaseEstimator
import numpy as np
import json
import pandas as pd
import os

# =================
# R packages import
# =================
import rpy2.robjects.packages as rpackages

# import R's utility package
utils = rpackages.importr('utils')

# select a mirror for R packages
utils.chooseCRANmirror(ind=1) # select the first mirror in the list

# required R package names to install if needed
packnames = ('ks')

# R vector of strings
from rpy2.robjects.vectors import StrVector

# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

# import ks
from rpy2.robjects.packages import importr
ks = importr('ks')

# activate numpy conversion for input arguments
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

# ====================
# R packes import done
# ====================

class KernelDensity(BaseEstimator):
    def __init__(self,
                 H=None,
                 bandwidth_selection=None,
                 diag=True,
                 dx=None,
                 grid_size=None,
                 bounded_features=[],
                 binned=False,
                 verbose=False):
        """
        Kernel density estimation for 1 to 6 dimensional data.
        The kernel is Gaussian.

        Parameters
        ----------
        H : array-like of shape (s,s), default=None
            The bandwidth matrix. If None, the bandwidth selection cannot be None.

        bandwidth_selection : {None, 'SCV', 'UCV', 'Pi'}, default=None
            None : No bandwidth selection is made and H is expected.

            SCV : Smoothed Cross-Validation method. If H is not None, H is considered as the initial bandwidth matrix.

            UCV : Unbiased Cross-Validation method. If H is not None, H is considered as the initial bandwidth matrix.

            Pi : Plug-in method. If H is not None, H is considered as the initial bandwidth matrix.

        diag : boolean, default=True,
            If True, the bandwidth selection is constrained to diagonal matrix.

        dx : array-like of shape (n_features,), default=None
            The expected grid step for each features. It overrides the grid_size

        grid_size : tuple of length s, default=None
            Grid sizes for each dimensions. If None, arbitrary grid sizes are set.
            The grid size is overrided by dx.

        bounded_features : list of int, default=[]
            Features indices which are bounded on low values.

        binned : bool, default=False
            If True, kernel estimation is approximated through binned data.
            It is usefull for large cases.

        verbose : bool, default=False
            If True, print out progress information

        """
        self.bandwidth_selection = bandwidth_selection
        self.H = H
        self.dx = dx
        self.grid_size = grid_size
        self.diag = diag
        self._selected_H = None
        self.bounded_features = bounded_features
        self.binned = binned
        self.verbose = verbose

    def fit(self,
            X,
            y=None,
            sample_weight=None,
            bandwidth_select=True):
        """
        Fit the Kernel Density model on the data

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row corresponds to a single data point.

        y : None
            ignored. This parameter exists only for compatibility with Sklearn Pipeline.

        sample_weight : array-like of shape (n_samples,), default=None
            List of sample weights attached to the data X.

        bandwidth_select : bool, default=True
            If True, the bandwidth selection process is made.

        Returns
        -------
        self : object
            Returns instance of object.
        """

        # mirror datas for bounded features
        X = X.copy()

        # computes grid size if dx is not None:
        if self.dx is not None:
            X_min = X.min(axis=0)
            X_max = X.max(axis=0)

            self.grid_size = np.round((X_max - X_min) / self.dx, 0) + 1

            self.grid_size[self.bounded_features] *= 2
            self.grid_size = self.grid_size.astype(int)

        # first, create data mirror in case of bounded columns
        self._low_bounds = X[:, self.bounded_features].min(axis=0)

        for idx, feature in enumerate(self.bounded_features):
            X_mirrored = X.copy()
            X_mirrored[:, feature] = 2 * self._low_bounds[idx] - X[:, feature]

            X = np.vstack((X, X_mirrored))

            if sample_weight is not None:
                sample_weight = np.vstack((sample_weight, sample_weight))

        self._data = X
        if sample_weight is not None:
            self._data_weights = sample_weight
        else:
            self._data_weights = None

        # Bandwidth selection
        if self.bandwidth_selection is not None and bandwidth_select:

            if self.bandwidth_selection == 'SCV':
                if self.diag :
                    func = ks.Hscv_diag
                else:
                    func = ks.Hscv
                # if only one feature
                if self._data.shape[1] == 1:
                    func = ks.hscv

            elif self.bandwidth_selection == 'UCV':
                if self.diag :
                    func = ks.Hlscv_diag
                else:
                    func = ks.Hlscv

                # if only one feature
                if self._data.shape[1] == 1:
                    func = ks.hlscv

            elif self.bandwidth_selection == 'Pi':
                if self.diag :
                    func = ks.Hpi_diag
                else:
                    func = ks.Hpi

                # if only one feature
                if self._data.shape[1] == 1:
                    func = ks.hpi

            args = dict(x = self._data,
                        binned = self.binned,
                        verbose = self.verbose)

            if self.H is not None and self._data.shape[1] > 1:
                args['Hstart'] = self.H

            if self.grid_size is not None:
                 args['bgridsize'] = np.array(self.grid_size)

            if self._data.shape[1] > 1:
                self._selected_H = np.array(func(**args))
            else:
                self._selected_H = np.array(func(**args))[0]

        return(self)

    def density(self, X):
        """
        Compute the estimated density.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            An array of points to query. Last dimension should match dimension of training data (n_features).

        Returns
        -------

        density : array of shape (n_samples,)
            The array of density evaluations.
        """

        if self._selected_H is not None:
            H = self._selected_H
        else:
            H = self.H

        args = dict(x=self._data,
                    eval_points=X,
                    binned = self.binned,
                    verbose = self.verbose)

        if self._data.shape[1] > 1:
            args['H'] = H
        else:
            args['h'] = H

        if self.grid_size is not None:
            args['gridsize'] = np.array(self.grid_size)

        if self._data_weights is not None:
            args['w'] = self._data_weights

        p = np.array(ks.kde(**args)[2])

        # set to 0 all values outside the X bounds
        idx = np.any(X[:, self.bounded_features] < self._low_bounds, axis=1)

        p[idx] = 0

        # multiply the y-values to get integral of ~1
        p = p * 2 ** len(self.bounded_features)

        return(p)

    def grid_density(self):
        """
        Get the grid density

        Returns
        -------
        X_grid : array-like of shape (n_grid, n_features)
            The grid.

        density : array-like of shape (n_grid,)
            The array of density evaluations.
        """
        if self._selected_H is not None:
            H = self._selected_H
        else:
            H = self.H

        args = dict(x = self._data,
                    binned = self.binned,
                    verbose = self.verbose)

        if self._data.shape[1] > 1:
            args['H'] = H
        else:
            args['h'] = H

        if self.grid_size is not None:
            args['gridsize'] = np.array(self.grid_size)

        if self._data_weights is not None:
            args['w'] = self._data_weights

        result = ks.kde(**args)

        if self._data.shape[1] > 1:
            xx = result[1]

            cols = np.meshgrid(*[np.array(xxi) for xxi in xx], indexing='ij')

            X_grid = np.vstack(tuple(col.flat for col in cols)).T

            p = np.array(result[2]).flat

        else:
            X_grid = np.array(result[1])
            X_grid = X_grid[:,None]

            p = np.array(result[2])

        # remove all values outside bounds
        idx = np.all(X_grid[:, self.bounded_features] >= self._low_bounds, axis=1)

        X_grid = X_grid[idx]
        p = p[idx]

        # multiply the y-values to get integral of ~1
        p = p * 2 ** len(self.bounded_features)

        return(X_grid, p)

    def save(self, path, data=False):
        """
        Export the estimation model in json format for parameters and in csv format for data, and bandwidth matrix.

        Parameters
        ----------
        path : str
            file path. Should be a zip file.

        data : bool, default=False
            if True, the data is also exported.
        """

        # params
        files_names = []
        folder_name = os.path.dirname(path)

        params = dict(
                        bandwidth_selection=self.bandwidth_selection,
                        dx=self.dx.tolist(),
                        grid_size=self.grid_size.tolist(),
                        diag=self.diag,
                        bounded_features=self.bounded_features,
                        binned=self.binned,
                        _low_bounds = self._low_bounds.tolist(),
                        )
        print(params)
        with open('params.json', 'w') as f:
            json.dump(params, f)

        files_names.append('params.json')

        # bandwidth matrix
        if self.H is not None:
            df = pd.DataFrame(self.H)
            df.to_csv("H.csv", index=False, header=False)
            files_names.append('H.csv')

        if self._selected_H is not None:
            df = pd.DataFrame(self._selected_H)
            df.to_csv("selected_H.csv", index=False)
            files_names.append('selected_H.csv')

        # data
        if data:
            df = pd.DataFrame(self._data)

            if self._data_weights is not None:
                df['weight'] = self._data_weights

            df.to_csv("data.csv", index=False, header=False)
            files_names.append('data.csv')

        # zip file
        command = 'zip .temp_export_file.zip'
        for file_name in files_names:
            command += ' ' + file_name
        os.system(command)

        command = 'rm '
        for file_name in files_names:
            command += ' ' + file_name
        os.system(command)

        command = 'mkdir -p ' + folder_name
        os.system(command)

        os.system('mv .temp_export_file.zip ' + path)


    def load(self, path):
        os.system('unzip ' + path + ' -d ' + path + '.out')

        files = os.listdir(path + '.out/')

        self.layers = {}
        for file in files:
            if file == 'params.json':
                f = open(path + '.out/params.json')
                params = json.load(f)

                for key, param in params.items():
                    setattr(self, key, param)

                f.close()

            if file == 'H.csv':
                df = pd.read_csv(path + '.out/H.csv', header=0)
                self.H = df.values.copy()

            if file == 'selected_H.csv':
                df = pd.read_csv(path + '.out/selected_H.csv', header=0)
                self._selected_H = df.values.copy()

            if file == 'data.csv':
                df = pd.read_csv(path + '.out/data.csv', header=1)
                if 'weight' in df.columns.to_list():
                    self._data_weights = df['weight']

                    df = df.drop('weight', 1)

                self._data = df.values

        os.system('rm -R ' + path + '.out')

