"""Density Estimation Methods"""


from ._gaussian_kde import GKDE
# from ._ash import ASH
from fastash import ASH
# from ._uniform_kde import UKDE
# from ._fft_kde import FFTKDE

_methods = {'gkde':GKDE,
            'ash':ASH}
