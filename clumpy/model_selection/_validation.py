# -*- coding: utf-8 -*-
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score as sklearn_cross_val_score
from ..metrics import log_score

def cross_val_score(estimator,
                    X,
                    y,
                    cv=5,
                    n_jobs=None,
                    verbose=0):
    """
    Evaluate a score by cross-validation
    """
        
    scoring = make_scorer(score_func=log_score,
                    greater_is_better=True,
                    needs_proba=True)
    
    return(sklearn_cross_val_score(estimator = estimator,
                            X = X,
                            y = y,
                            cv = StratifiedKFold(n_splits=cv,
                                                 shuffle=True),
                            scoring=scoring,
                            verbose=verbose,
                            n_jobs=n_jobs))
    