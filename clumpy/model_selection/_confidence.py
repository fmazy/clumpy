import numpy as np
from mahalanobis import Mahalanobis
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm
from sklearn.model_selection import KFold

def cross_reject(estimator,
                 X,
                 cv=5,
                 shuffle=True,
                 alpha=0.9,
                 distance='ks',
                 n_mc=10**3,
                 n_jobs=1):

    if type(cv) is int:
        kf = KFold(n_splits = cv,
                   shuffle = shuffle)

        train_indices = []
        test_indices = []
        for train_index, test_index in kf.split(X):
            train_indices.append(train_index)
            test_indices.append(test_index)
    else:
        train_indices = cv[0]
        test_indices = cv[1]

    r = []
    for i_cv in range(len(train_indices)):
        X_train = X[train_indices[i_cv]]
        X_test = X[test_indices[i_cv]]

        estimator.fit(X_train)

        r.append(reject(estimator = estimator,
                        X_test = X_test,
                        alpha = alpha,
                        distance = distance,
                        n_mc = n_mc,
                        n_jobs = n_jobs))

    return(r)

def _ks_distance(X1, X2):
    return(np.max(np.abs(np.power(X1-X2,1))))

def reject(estimator, X_test, alpha=0.9, distance='ks', n_mc=10**3, n_jobs=1):
    # distance function selector
    if distance == 'ks':
        distance_func = _ks_distance

    # first get G for train data :
    print('get model distribution')
    mah = Mahalanobis(estimator.data, calib_rows=-1)
    delta = np.sort(mah.calc_distances(estimator.data)[:, 0])

    delta_test = np.sort(mah.calc_distances(X_test)[:, 0])

    # ici, on calcule G aux points de delta, i.e aux points de X_train
    # on pourrait regarder aux points de X_test aussi.
    # à tester.
    G = _edf(delta, delta)
    G_test = _edf(delta, delta_test)

    d_test = distance_func(G, G_test)

    d_mc = _delta_train_distribution(estimator,
                                     n_test = X_test.shape[0],
                                     mah = mah,
                                     delta = delta,
                                     G = G,
                                     distance_func = distance_func,
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



def _delta_train_distribution(estimator, n_test, mah, delta, G, distance_func=_ks_distance, n_mc=10**3, n_jobs=1, sort=True):
    pool = Pool(n_jobs)
    d_mc = pool.uimap(_compute_distance_to_train,
                      [estimator for i in range(n_mc)],
                      [distance_func for i in range(n_mc)],
                      [n_test for i in range(n_mc)],
                      [mah for i in range(n_mc)],
                      [delta for i in range(n_mc)],
                      [G for i in range(n_mc)])

    d_mc = np.array(list(d_mc))

    if sort:
        d_mc = np.sort(d_mc)

    return(d_mc)


def _compute_distance_to_train(estimator, distance_func, n_test, mah, delta, G):
    X_sample = estimator.sample(n_test, exact=False)

    # mah_sigma = Mahalanobis(X_sample, calib_rows=-1)
    delta_sample = mah.calc_distances(X_sample)[:, 0]

    # on évalue G_sigma aux mêmes points que G
    G_sample = _edf(delta, delta_sample)

    return (distance_func(G, G_sample))


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


