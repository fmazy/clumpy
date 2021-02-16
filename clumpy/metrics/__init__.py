"""The :mod:`clumpy.metrics` module completes :mod:`sklearn.metrics`. """

from ._log_score import log_score, compute_a, log_scorer, under_sampling_log_scorer
from ._weighted_kolmogorov_smirnov import weighted_kolmogorov_smirnov
from ._mcdf import mcdf