"""Density Estimation Methods"""


from ._gaussian_kde import GKDE
from . import bandwidth_selection
from ._whitening_transformer import _WhiteningTransformer
from ._density_estimator import DensityEstimationParams

_methods = {'gkde':GKDE,}