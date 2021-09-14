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
   
:mod:`clumpy.kde`: Kernel Density Estimation
============================================

.. automodule:: clumpy.kde
    :no-members:
        :no-inherited-members:

.. currentmodule:: clumpy

.. autosummary::
   :toctree: generated/
   :template: class.rst
    
   kde.GKDE
   
:mod:`clumpy.allocation`: Allocation
====================================

.. automodule:: clumpy.allocation
    :no-members:
    :no-inherited-members:
    
.. currentmodule:: clumpy

.. autosummary::
   :toctree: generated/
   :template: function.rst
    
   allocation.generalized_allocation
  
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
   resampling.under_sampling.correct_probabilities
   
:mod:`clumpy.scenario`: Scenario
================================

.. automodule:: clumpy.scenario
    :no-members:
    :no-inherited-members:
    
.. currentmodule:: clumpy

.. autosummary::
   :toctree: generated/
   :template: function.rst
    
   scenario.adjust_probabilities

:mod:`clumpy.resampling`: utils
===============================

.. automodule:: clumpy.utils
    :no-members:
    :no-inherited-members:
    
.. currentmodule:: clumpy

.. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.human_size
   utils.ndarray_suitable_integer_type
