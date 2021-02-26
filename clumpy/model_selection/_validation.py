# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler

def cross_val_score(estimator,
                    X,
                    cv=5,
                    standard_scaler=True,
                    n_jobs=1,
                    verbose=0):
    """
    Evaluate a score by cross-validation
    """

    if type(cv) == int:
        cv = ShuffleSplit(n_splits=cv)
        cv.split(X)

    scores = []

    i_cv = 0
    for train_index, test_index in cv:
        i_cv += 1
        if verbose>0:
            print('cv #'+str(i_cv)+'...')
        X_train = X[train_index, :]
        X_test = X[test_index, :]

        if standard_scaler:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        estimator.fit(X_train)
        scores.append(estimator.score(X_test,
                                      n_jobs=n_jobs,
                                      verbose=verbose - 1))

        if verbose>0:
            print('score: '+str(scores[-1]))

    return(np.array(scores))
