# -*- coding: utf-8 -*-
from ._split import StratifiedKFold


def cross_val_score(estimator,
                    X_u,
                    v_u,
                    cv=5,
                    verbose=0):
    """
    Evaluate a score by cross-validation
    """
    
    if type(cv) == int:
        cv = StratifiedKFold(n_splits=cv)
        
    elif not isinstance(cv, StratifiedKFold):
        raise ValueError("cv type incorrect. Expected StratifiedKFold class object")
    
    # stratified K fold
    train_index_u, test_index_u = cv.split(X_u, v_u)
    
    scores = []
    
    # for each split
    for n in range(cv.n_splits):
        if verbose > 0:
            print('split #'+str(n))
        # first create train test for this split
        X_u_train = {}
        v_u_train = {}
        X_u_test = {}
        v_u_test = {}
        
        for u in X_u.keys():
            X_u_train[u] = X_u[u][train_index_u[u][n],:]
            v_u_train[u] = v_u[u][train_index_u[u][n]]
            X_u_test[u] = X_u[u][test_index_u[u][n],:]
            v_u_test[u] = v_u[u][test_index_u[u][n]]
        
        # then train estimator
        estimator.fit(X_u_train, v_u_train)
        
        # get the score and append it to scores
        scores.append(estimator.score(X_u_test, v_u_test))
    
    return(scores)
    