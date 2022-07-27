CLUMPY

*Comprehensive Land Use Models in Python*

**Keywords :** Land use and cover change, LUCC, GIS, geostatistic, spatially explicit model.

**Functional Description :**
CLUMPY is a land use and cover change model which aims to be used in environmental science in a variety of contexts. It allows to observe past changes statistics, set future contrasted scenarios and allocate simulated future land use maps according to those projections.
From past land use maps and some explanatory variables, the software calibrates the model through machine learning methods (kernel density estimations). Explicit probabilistic values are returned and can be then adjusted by the user to set contrasted scenarios. It is finally possible to allocate a simulated land use map in order to provide decision materials for public stakeholder.
Other land use and cover change model softwares such as Dinamica EGO, CLUE and LCM are compared to CLUMPY in papers to come.

**Project Management :**
CLUMPY is in Beta version for now. The first stable and fully working version is planed for October 2022. This version 

**References :**
* A Formally Correct and Algorithmically Efficient LULC Change Model-building Environment, Mazy, F.-R. and Longaretti, P.-Y., 2022, Proceedings of the 8th International Conference on Geographical Information Systems Theory, Applications and Management - GISTAM, p. 25-36, INSTICC, ScitePress, doi: 10.5220/0011000000003185
* Towards a Generic Theoretical Framework for spatially explicit statistical LUCC Modeling
    * An accurate and powerful calibration-estimation method based on Kernel density estimation, Mazy, F.-R. and Longaretti, P.-Y., 2022, Submitted to Environmental Software and Modeling
    * Allocation Revisited: Formal foundations and bias identification, Mazy, F.-R. and Longaretti, P.-Y., 2022, Planed to be submitted to Environmental Software and Modeling
    * A maximum relevance / minimum redundancy selection procedure of explanatory variables, Mazy, F.-R. and Longaretti, P.-Y., 2022, Planed to be submitted to Environmental Software and Modeling

**Participants :**
* François-Rémi Mazy \(^1\) (dev)
* Pierre-Yves Longaretti \(^{1,2}\)

1: Université Grenoble Alpes, CNRS, Inria, Grenoble INP, LJK, 38000 Grenoble, France
2 : Université Grenoble Alpes, CNRS-INSU, IPAG, CS 40700, 38052 Grenoble, France

**Contact :**
François-Rémi Mazy, francois-remi.mazy@inria.fr

## Install

*The software actually provided by pip is not up to date !*

Create conda env including gdal and rasterio
`conda create -n clumpy python=3.8 gdal rasterio`

build the package (see 'setup.py' for requirements information) :
`python3 setup.py sdist bdist_wheel`

