##########
Clumpy API
##########

This is the full API documentation of the `clumpy` toolbox.

Base objects
============

.. automodule:: clumpy._base
    :no-members:
    :no-inherited-members:
    
.. currentmodule:: clumpy

State and palette
-----------------

.. automodule:: clumpy._base._state
   :no-members:
   :no-inherited-members:
   
.. currentmodule:: clumpy

.. autosummary::
   :toctree: generated/
   :template: class.rst
    
   State
   Palette

Layers
------

.. automodule:: clumpy._base._layer
   :no-members:
   :no-inherited-members:

.. currentmodule:: clumpy

.. autosummary::
   :toctree: generated/
   :template: class.rst

   LandUseLayer
   MaskLayer
   FeatureLayer

Levels
------
.. autosummary::
   :toctree: generated/
   :template: class.rst

   Land
   Region
   Territory

Transition matrix
-----------------
.. autosummary::
   :toctree: generated/
   :template: class.rst

   TransitionMatrix

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_transition_matrix
   load_transition_matrix

Estimation
==========
Density Estimation
------------------

.. automodule:: clumpy.density_estimation
   :no-members:
   :no-inherited-members:

.. currentmodule:: clumpy

.. autosummary::
   :toctree: generated/
   :template: class.rst

   density_estimation.GKDE

Transition Probability Estimation
---------------------------------
.. automodule:: clumpy.transition_probability_estimation
   :no-members:
   :no-inherited-members:

.. currentmodule:: clumpy

.. autosummary::
   :toctree: generated/
   :template: class.rst

   transition_probability_estimation.Bayes

Allocation
==========
Patches
-------
.. automodule:: clumpy.allocation._patch
   :no-members:
   :no-inherited-members:

.. currentmodule:: clumpy

.. autosummary::
   :toctree: generated/
   :template: class.rst

   allocation.BootstrapPatch

.. autosummary::
   :toctree: generated/
   :template: function.rst

   allocation.compute_bootstrap_patches

Allocators
----------
.. automodule:: clumpy.allocation._allocator
   :no-members:
   :no-inherited-members:

.. currentmodule:: clumpy

.. autosummary::
   :toctree: generated/
   :template: class.rst

   allocation.Unbiased
