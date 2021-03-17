import numpy as np
from mahalanobis import Mahalanobis
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm
from sklearn.model_selection import KFold

class ParameterGrowthReject():
    def __init__(self,
                 estimator,
                 param_key,
                 param_bounds,
                 epsilon=0.05,
                 cv=5,
                 alpha=0.95,
                 distance='ks',
                 n_mc=10**3,
                 n_jobs=1,
                 verbose=0):

        self.estimator = estimator
        self.param_key = param_key
        self.param_bounds = param_bounds
        self.epsilon = epsilon
        self.cv = cv
        self.alpha = alpha
        self.distance = distance
        self.n_mc = n_mc
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X):
        if type(self.cv) is int:
            kf = KFold(n_splits=self.cv,
                       shuffle=True)

            train_indices = []
            test_indices = []
            for train_index, test_index in kf.split(X):
                train_indices.append(train_index)
                test_indices.append(test_index)

            cv = [train_indices, test_indices]
        else:
            cv = self.cv

        # set attributes to estimator
        setattr(self.estimator, self.param_key, np.mean(self.param_bounds))


        param_min = self.param_bounds[0]
        param_max = self.param_bounds[1]
        param_last = np.mean(self.param_bounds) + self.epsilon * 10

        while np.abs(param_last - getattr(self.estimator, (self.param_key))) / param_last > self.epsilon:
            if self.verbose > 0:
                print(self.param_key + ' = ' + str(getattr(self.estimator, self.param_key)))
            r_cv = cross_reject(self.estimator,
                                X=X,
                                cv=cv,
                                alpha=self.alpha,
                                distance=self.distance,
                                n_mc=self.n_mc,
                                n_jobs=self.n_jobs,
                                verbose=self.verbose)

            param_last = getattr(self.estimator, self.param_key)

            # if one of them is rejected,
            # then the parameter is too high
            print(r_cv)
            if np.any(r_cv):
                print('too high')
                setattr(self.estimator, self.param_key, (param_min + param_last) / 2)
                param_max = param_last

            # else, the parameter is too low
            else:
                print('too low')
                setattr(self.estimator, self.param_key, (param_last + param_max) / 2)
                param_min = param_last

        # the while loop is done
        # the right parameter should be the last param
        # if the last test is failed, the param should be param_min
        if np.any(r_cv):
            setattr(self.estimator, self.param_key, param_min)
        else: # else, it's the param_last
            setattr(self.estimator, self.param_key, param_last)

        self.estimator.fit(X)


def cross_reject(estimator,
                 X,
                 cv=5,
                 alpha=0.95,
                 distance='ks',
                 n_mc=10**3,
                 n_jobs=1,
                 verbose=0):

    if type(cv) is int:
        kf = KFold(n_splits = cv,
                   shuffle = True)

        train_indices = []
        test_indices = []
        for train_index, test_index in kf.split(X):
            train_indices.append(train_index)
            test_indices.append(test_index)

        cv = [train_indices, test_indices]

    if verbose > 0:
        loop_cv = tqdm(range(len(cv[0])))
    else:
        loop_cv = range(len(cv[0]))

    r = []
    for i_cv in loop_cv:
        X_train = X[cv[0][i_cv]]
        X_test = X[cv[1][i_cv]]

        estimator.fit(X_train)

        r.append(reject(estimator = estimator,
                        X_test = X_test,
                        alpha = alpha,
                        distance = distance,
                        n_mc = n_mc,
                        n_jobs = n_jobs))

    return(r)

def _ks_distance(X1, X2):
    return(np.max(np.abs(X1-X2)))

def reject(estimator, X_test, alpha=0.9, distance='ks', n_mc=10**3, n_jobs=1):
    # distance function selector
    if distance == 'ks':
        distance = _ks_distance

    # first get G for train data :
    mah = Mahalanobis(estimator.data, calib_rows=-1)
    delta = np.sort(mah.calc_distances(estimator.data)[:, 0])

    delta_test = np.sort(mah.calc_distances(X_test)[:, 0])

    # ici, on calcule G aux points de delta, i.e aux points de X_train
    # on pourrait regarder aux points de X_test aussi.
    # à tester.
    G = _edf(delta, delta)
    G_test = _edf(delta, delta_test)

    d_test = distance(G, G_test)

    d_mc = _delta_train_distribution(estimator,
                                     n_test = X_test.shape[0],
                                     mah = mah,
                                     delta = delta,
                                     G = G,
                                     distance = distance,
                                     n_mc = n_mc,
                                     n_jobs = n_jobs,
                                     sort = True)

    # p-values according to alpha
    cdf_d_mc = d_mc.cumsum() / d_mc.sum()
    p_1 = d_mc[np.argmax(cdf_d_mc >= (1 - alpha) / 2)]
    p_2 = d_mc[np.argmax(cdf_d_mc >= 1 - (1 - alpha) / 2)]

    # is d_test between p-values ?
    if d_test > p_1 and d_test < p_2:
        return(False)

    return(True)

    # print('get bootstrap distribution')
    # d_b = pool.uimap(self._compute_bootstrap_distance,
    #                  [n_test for pi in range(n_b)],
    #                  [mah for pi in range(n_b)],
    #                  [delta for pi in range(n_b)],
    #                  [G for pi in range(n_b)])
    # d_b = np.sort(np.array(list(d_b)))

    # print('analysing')
    # # compute p values for alpha
    # cdf = d_mc.cumsum() / d_mc.sum()
    #
    # p_1 = d_mc[np.argmax(cdf >= (1 - alpha) / 2)]
    # p_2 = d_mc[np.argmax(cdf >= 1 - (1 - alpha) / 2)]
    #
    # # compute cdf for bootstraped distances
    # cdf_b = d_b.cumsum() / d_b.sum()
    #
    # cdf_b_at_p_1 = cdf_b[np.argmax(d_b >= p_1)]
    # cdf_b_at_p_2 = cdf_b[np.argmax(d_b >= p_2)]
    # if d_b.max() <= p_1:
    #     cdf_b_at_p_1 = 1
    # if d_b.max() <= p_2:
    #     cdf_b_at_p_2 = 1
    #
    # return (cdf_b_at_p_2 - cdf_b_at_p_1)



def _delta_train_distribution(estimator,
                              n_test,
                              mah,
                              delta,
                              G,
                              distance=_ks_distance,
                              n_mc=10**3,
                              n_jobs=1,
                              sort=True):
    pool = Pool(n_jobs)
    d_mc = pool.uimap(_compute_distance_to_train,
                      [estimator for i in range(n_mc)],
                      [distance for i in range(n_mc)],
                      [n_test for i in range(n_mc)],
                      [mah for i in range(n_mc)],
                      [delta for i in range(n_mc)],
                      [G for i in range(n_mc)])

    d_mc = np.array(list(d_mc))

    if sort:
        d_mc = np.sort(d_mc)

    return(d_mc)


def _compute_distance_to_train(estimator,
                               distance,
                               n_test,
                               mah,
                               delta,
                               G):
    X_sample = estimator.sample(n_test, exact=False)

    # mah_sigma = Mahalanobis(X_sample, calib_rows=-1)
    delta_sample = mah.calc_distances(X_sample)[:, 0]

    # on évalue G_sigma aux mêmes points que G
    G_sample = _edf(delta, delta_sample)

    return (distance(G, G_sample))


# def _compute_bootstrap_distance(self, n_test, mah, delta, G):
#     X_b = self.data[np.random.choice(a=self.data.shape[0],
#                                      size=n_test,
#                                      replace=True)]
#
#     delta_b = np.sort(mah.calc_distances(X_b)[:, 0])
#     G_b = _edf(delta, delta_b)
#
#     return (_d_func(G, G_b))

def _edf(X_eval, X_train):
    if len(X_train.shape) > 1:
        return(np.array([np.all(X_train <= x, axis=1).sum() for x in X_eval])/X_train.shape[0])
    else:
        return(np.array([(X_train <= x).sum() for x in X_eval])/X_train.size)


