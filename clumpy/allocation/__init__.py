from ._unbiased import Unbiased
from ._unbiased_mono_pixel import UnbiasedMonoPixel
from ._patch import BootstrapPatch
from ._compute_patches import compute_bootstrap_patches

methods = {'unbiased': Unbiased,
           'unbiased_mono_pixel': UnbiasedMonoPixel}
