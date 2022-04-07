from ._variance_threshold import VarianceThreshold
from ._correlation_threshold import CorrelationThreshold
from ._mRMR import MRMR

feature_selectors = {'variance_threshold':VarianceThreshold,
                     'correlation_threshold':CorrelationThreshold}
