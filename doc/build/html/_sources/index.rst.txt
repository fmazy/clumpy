.. clumpy documentation master file, created by
   sphinx-quickstart on Tue Jun  2 14:31:59 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
clumpy : Comprehensive Land Use Models in Python
================================================

Clumpy is a land use and cover change (LUCC) modeler in development written in Python3. Only basic parameters are for now avaiable through python script usage. A GUI is considered as a plugin of Qgis which uses this present library.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation
   tutorials
   release_notes

.. toctree::
    :maxdepth: 1
    :caption: Definition

    definition/layer
    definition/case

.. toctree::
    :maxdepth: 1
    :caption: Discretization

    discretization/discretization

.. toctree::
    :maxdepth: 1
    :caption: Calibration
    
    calibration/naive_bayes
    

.. toctree::
    :maxdepth: 1
    :caption: Allocation
    
    allocation/simple_unbiased
    allocation/generalized_von_neumann
    allocation/dinamica
    
    
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
