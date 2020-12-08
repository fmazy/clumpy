##########
Clumpy API
##########

This is the full API documentation of the `clumpy` toolbox.

:mod:`clumpy.definition`: Definition
====================================

.. automodule:: clumpy.definition
    :no-members:
    :no-inherited-members:
    
.. currentmodule:: clumpy

Layer
-----

.. automodule:: clumpy.definition._layer
   :no-members:
   :no-inherited-members:

.. currentmodule:: clumpy

.. autosummary::
   :toctree: generated/
   :template: class.rst
    
   definition.LandUseCoverLayer
   definition.FeatureLayer
   definition.DistanceToVFeatureLayer

Case
----

.. automodule:: clumpy.definition._case
   :no-members:
   :no-inherited-members:

.. currentmodule:: clumpy

.. autosummary::
   :toctree: generated/
   :template: class.rst
    
   definition.Case

:mod:`clumpy.resampling`: Resampling
====================================

.. automodule:: clumpy.resampling
    :no-members:
    :no-inherited-members:
    
.. currentmodule:: clumpy

Under sampling
--------------

.. automodule:: clumpy.resampling.under_sampling
   :no-members:
   :no-inherited-members:

.. currentmodule:: clumpy

.. autosummary::
   :toctree: generated/
   :template: function.rst
    
   resampling.under_sampling.compute_sampling_strategy
   resampling.under_sampling.log_scorer_corrected
   
:mod:`clumpy.metrics`: Metrics
==============================

.. automodule:: clumpy.metrics
    :no-members:
    :no-inherited-members:
    
.. currentmodule:: clumpy

Log score
---------

.. automodule:: clumpy.metrics._log_score
   :no-members:
   :no-inherited-members:

.. currentmodule:: clumpy

.. autosummary::
   :toctree: generated/
   :template: function.rst
    
   metrics.compute_a
   metrics.log_score
   metrics.log_scorer
   
:mod:`clumpy.scenario`: Scenario
===============================

.. automodule:: clumpy.scenario
    :no-members:
    :no-inherited-members:
    
.. currentmodule:: clumpy

Probabilities
-------------

.. automodule:: clumpy.scenario._probabilities
   :no-members:
   :no-inherited-members:

.. currentmodule:: clumpy

.. autosummary::
   :toctree: generated/
   :template: function.rst
    
   scenario.adjust_probabilities

   
