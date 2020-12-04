from ._cluster import _Cluster

from sklearn.cluster import OPTICS as OPTICSsklearn
from sklearn.neighbors import NearestNeighbors
import numpy as np

class OPTICS(_Cluster):
    def __init__(self, min_samples=2, max_eps=np.inf, k=0.2):
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.k = k
        
    def _new_estimator(self):
        return(_OPTICSEstimator(min_samples = self.min_samples,
                                max_eps = self.max_eps,
                                k=self.k))
    
class _OPTICSEstimator():
    def __init__(self, min_samples=2, max_eps=np.inf, k=0.2):
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.k = k
    
    def fit(self, X):
        
        if self.k <= 1:
            n_fit = int(X.shape[0]*self.k)
        else:
            n_fit = self.k
        
        print('agglomerative clustering')
        optics = OPTICSsklearn(min_samples=self.min_samples,
                               max_eps=self.max_eps)
        
        self.X_sample = X[np.random.choice(np.arange(X.shape[0]),
                                       size=n_fit,
                                       replace=False),:]
        
        self.labels_sample = optics.fit_predict(self.X_sample)
        
        self.neigh = NearestNeighbors(n_neighbors=1)
        self.neigh.fit(self.X_sample)
            
    def predict(self, X):
        """
        predict :math:`P(z|v_i,v_f)`

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        return_labels : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        return(self.labels_sample[self.neigh.kneighbors(X, n_neighbors=1, return_distance=False)[:,0]].reshape(-1,1))