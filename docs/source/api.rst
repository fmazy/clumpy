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

State
-----

.. automodule:: clumpy.definition._state
   :no-members:
   :no-inherited-members:
   
.. currentmodule:: clumpy

.. autosummary::
   :toctree: generated/
   :template: class.rst
    
   definition.State
   definition.Palette

Layer
-----

.. automodule:: clumpy.definition._layer
   :no-members:
   :no-inherited-members:

.. currentmodule:: clumpy

.. autosummary::
   :toctree: generated/
   :template: class.rst
    
   definition.LandUseLayer
   definition.FeatureLayer
   
Region
------
.. automodule:: clumpy.definition._region
   :no-members:
   :no-inherited-members:

.. currentmodule:: clumpy

.. autosummary::
   :toctree: generated/
   :template: class.rst
    
   definition.Region

Land
----
.. automodule:: clumpy.definition._land
   :no-members:
   :no-inherited-members:

.. currentmodule:: clumpy

.. autosummary::
   :toctree: generated/
   :template: class.rst
    
   definition.Land
   
Patches
-------
.. automodule:: clumpy.definition._patches
   :no-members:
   :no-inherited-members:

.. currentmodule:: clumpy

.. autosummary::
   :toctree: generated/
   :template: class.rst

   definition.BootstrapPatches
   definition.TransitionPatches

:mod:`clumpy.definition`: Scenario
==================================

.. automodule:: clumpy.scenario
    :no-members:
    :no-inherited-members:
    
.. currentmodule:: clumpy

Transition Matrix
-----------------

.. automodule:: clumpy.scenario._transition_matrix
   :no-members:
   :no-inherited-members:
   
.. currentmodule:: clumpy

.. autosummary::
   :toctree: generated/
   :template: class.rst
    
   scenario.TransitionMatrix
   
.. autosummary::
    :toctree: generated/
    :template: function.rst

    scenario.compute_transition_matrix
    scenario.load_transition_matrix

:mod:`clumpy.density_estimation`: Density Estimation
====================================================

.. automodule:: clumpy.density_estimation
    :no-members:
    :no-inherited-members:
    
.. currentmodule:: clumpy


.. autosummary::
   :toctree: generated/
   :template: class.rst
    
   density_estimation.GKDE
   density_estimation.Parameters

:mod:`clumpy.models`: Models
==========================

.. automodule:: clumpy.models
    :no-members:
    :no-inherited-members:
    
.. currentmodule:: clumpy

Parameters
----------

.. automodule:: clumpy.models.parameters
    :no-members:
    :no-inherited-members:

.. currentmodule:: clumpy

.. autosummary::
    :toctree: generated/
    :template: class.rst
        
    models.PatchesParams
    models.BootstrapPatchParams
    models.LandParams
    models.RegionParams
        
Models functions
----------------

.. currentmodule:: clumpy

.. autosummary::
    :toctree: generated/
    :template: function.rst
    
    models.allocation
..
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
