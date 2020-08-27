"""
intro
"""

from . import patches
from . import feature_selection
from ._knn import KNeighborsRegressor
from ._kde import KernelDensity
from ._naive_bayes import NaiveBayes
from .train_test_split import train_test_split
from ._calibration import compute_P_vf__vi, compute_P_z__vi, compute_P_z__vi_vf, compute_P_vf__vi_z, compute_P_vf__vi_z_with_bayes