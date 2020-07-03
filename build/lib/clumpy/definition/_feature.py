# external libraries
"""
Explanatory variables are function used to characterise land use changes.
"""

import numpy as np
from matplotlib import pyplot as plt
#import pandas as pd

#internal libraries
from ._layer import DistanceToVFeatureLayer

class _Zk(object):
    """
    Explanatory variable object used for a fixed initial state :math:`v_i` whose data is based on :class:`.LayerEV`. This object is then used for the calibration and the allocation.
    
    This object is expected to be called through the functions :
    
        * :func:`.addZkFromLayer`
        * :func:`.addZkAsDistanceToV`
    """
    def __init__(self, name, kind, Ti):
        self.name = name
        self.kind = kind
        self.Ti = Ti
        
        self.param_discretization = {}
        
        self.param_calibration = {'smooth':None}
        
        # self.min = np.min(self.layer_EV.data.flat[self.Ti.J_vi.index.values])
        # self.max = np.max(self.layer_EV.data.flat[self.Ti.J_vi.index.values])
        # self.disc1rete = discrete
        # self.dyn = dyn
#         self.enabled = True
# #        self.J = []
        
#         # bins parameters
#         self.delta = 100
#         self.eta = 0.001
        
    def hist(self, bins=20, density=False):
        """
        displays an histogram of all values taken by :math:`Z_k`.
        
        :param bins: bins parameter, see `numpy documentation <https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges>`_ -- Default value : ``'auto'``.
        :param density: if True, the histogram is normalized, i.e. the area under the histogram will sum to 1 -- Default value : ``'auto'``.
        :type density: bool, optional
        """
        plt.hist(self.layer_EV.data.flat[self.Ti.J_vi.index.values], bins=bins, density=density)
        
    def histBins(self):
        """
        displays a bar plot of bins defined for :math:`Z_k`.
        """
        alpha = self.Ti.T.Z_alpha.loc[(self.Ti.T.Z_alpha.k==self.k) &
                                      (self.Ti.T.Z_alpha.vi==self.Ti.vi)].alpha.values
        
        coef = np.sum(self.alpha_N * np.diff(alpha))
        
        plt.bar(x=alpha[:-1],
                height=self.alpha_N/coef,
                width=np.diff(alpha),
                align='edge')
    
    # def setCalibrationParameters(self, smooth='DoNotChange'):
    #     if smooth != 'DoNotChange':
    #         self.calibration_parameters['smooth'] = smooth
        
    def __str__(self):
        txt = self.name+"\n"
        txt += "[min, max]: "+str(self.min)+", "+str(self.max)
        return(txt)
        
def _addZkAsDistanceToV(Ti, v, k=None, dyn=True):
    """
    Add an explanatory variable for an initial state ``Ti`` as a distance from :math:`v_i` to a fixed state :math:`v`. The new function is then appended to the ``Ti.Z`` list::
        
        dm.definition.explanatory_variable.addZkAsDistanceToV(T1, 2, "dist from 1 to 2")
    
    :param Ti: transition object with :math:`v_i` fixed
    :type Ti: _transition._Transition_vi
    :param v: targeted state
    :type v: int
    :param k: explanatory variable key -- Default : `None`, i.e. the name `distance_vix_to_vx` is used
    :type k: str, optional
    :param dyn: if `True`, the distance is updated at each time step during the allocation stage. 
    :type dyn: bool, optional

    :returns: a :class:`.Zk` object
    """
    # contrôle, la fonction existe déjà ??
    print("adding distance from "+str(Ti.vi)+" to "+str(v))
    
    # looking if it has already been computed
    dist = None
    for d in Ti.T.map_i.distance2v:
        if d.id_v == v:
            dist = d
    if dist == None:
        print("distance does not exist. Computing...")
        dist = DistanceToVFeatureLayer(id_v=v, layer_LUC=Ti.T.map_i)
        
    print("adding Zk")
    
    # vérification : Zk existe déjà ?
    if k == None:
        k = "distance_vi"+str(Ti.vi)+"_to_v"+str(v)
        
    if k in Ti.Z.keys():
        print("ERROR: this EV key is already used")
        return("ERROR")
    
    Ti.Z[k] = _Zk(k = k,
                 kind="distance2v",
                 layer_EV = dist,
                 Ti=Ti,
                 discrete=False,
                 dyn=dyn)
    
    print("\t done")
    return(Ti.Z[k])
    
def _addZkFromLayer(Ti, layer_EV, k, discrete=False):
    """
    Add an explanatory variable for an initial state ``Ti`` based on a layer. The new function is then appended to the ``Ti.Z`` list::
        
        dm.definition.explanatory_variable.addZkFromLayer(T1, dem, "elevation")
    
    :param Ti: transition object with :math:`v_i` fixed
    :type Ti: _transition._Transition_vi
    :param layer_EV: Layer to use as an explanatory variable
    :type layer_EV: :class:`.layer.LayerEV`
    :param k: explanatory variable key
    :type k: str
    :param discrete: bool, optional
    :type discrete: if ``True``, the explanatory variable is considered as discrete -- default : ``False``.

    :returns: a :class:`.Zk` object
    """
    
    # contrôle, la fonction existe déjà ??
    print("adding new explain factor from "+layer_EV.name)
    
    if k in Ti.Z.keys():
        print("ERROR: this EV key is already used")
        return("ERROR")
    
    Ti.Z[k] = _Zk(k = k,
             kind="static_EF_fromMap",
             layer_EV = layer_EV,
             Ti=Ti,
             discrete=discrete,
             dyn=False)

    print("\t done")
    return(Ti.Z[k])
