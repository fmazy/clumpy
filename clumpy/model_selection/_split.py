# -*- coding: utf-8 -*-

from ..utils import check_list_parameters_vi

from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.model_selection import StratifiedKFold as Sklearn_StratifiedKFold
from sklearn.utils import indexable

def train_test_split(*dicts, **options):
    """
    Split arrays or matrices into random train and test subsets for each initial states vi.
    
    Based on the sklearn function : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html?highlight=split#sklearn.model_selection.train_test_split

    Parameters
    ----------
    dicts : dicts of array-likes of shape (n_samples, n_features)
        
    test_size_vi : dict of float or int, default=None
        Dict of test_size for each initial state vi. If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the default value for each initial state will be set to 0.25 (i.e. the sklearn default value).

    Returns
    -------
        X_train_vi, X_test_vi, y_train_vi, y_test_vi.

    """
    n_arrays = len(dicts)
    if n_arrays == 0:
        raise ValueError("At least one dict of arrays required as input")
    
    test_size_vi = options.pop('test_size_vi', None)
    
    dicts = indexable(*dicts)
    
    check_list_parameters_vi(dicts)
    classes = dicts[0].keys()
    
    if test_size_vi is not None:
        print(test_size_vi)
        check_list_parameters_vi([dicts[0], test_size_vi])
            
    else:
        test_size_vi = {}
        for vi in classes:
            test_size_vi[vi] = None
    
    
    splitted = {}
    for vi in classes:
        splitted[vi] = sklearn_train_test_split(*(d[vi] for d in dicts), test_size=test_size_vi[vi])
        
    splitted_dicts = []
    for i in range(n_arrays*2):
        sd = {}
        for vi in classes:
            sd[vi] = splitted[vi][i]
        splitted_dicts.append(sd)
        
    return(splitted_dicts)

class StratifiedKFold():
    """Stratified K folds cross-validator"""
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
    def split(self, X_u, v_u):
        train_index_u = {}
        test_index_u = {}
        
        for u in X_u.keys():
            
            skf = Sklearn_StratifiedKFold(n_splits = self.n_splits,
                                          shuffle = self.shuffle,
                                          random_state = self.random_state)
            
            train_index_u[u] = []
            test_index_u[u] = []
            
            for train_index, test_index in skf.split(X_u[u], v_u[u]):
                train_index_u[u].append(train_index)
                test_index_u[u].append(test_index)
        
        return(train_index_u, test_index_u)
        
        
        
        