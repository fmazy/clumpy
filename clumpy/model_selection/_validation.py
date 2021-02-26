# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import ShuffleSplit

def cross_val_score(estimator,
                    X,
                    n_splits=5,
                    n_jobs=1,
                    verbose=0):
    """
    Evaluate a score by cross-validation
    """

    rs = ShuffleSplit(n_splits=n_splits)

    scores = []

    i_cv = 0
    for train_index, test_index in rs.split(X):
        i_cv += 1
        if verbose>0:
            print('cv #'+str(i_cv)+'...')
        X_train = X[train_index, :]
        X_test = X[test_index, :]

        estimator.fit(X_train)
        scores.append(estimator.score(X_test,
                                      n_jobs=n_jobs,
                                      verbose=verbose - 1))

        if verbose>0:
            print('score: '+str(scores[-1]))

    return(np.array(scores))
