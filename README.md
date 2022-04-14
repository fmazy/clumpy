# Clumpy

*Comprehensive Land Use Models in Python*

**Keywords :** Land use and cover change, LUCC, GIS, geostatistic, spatially explicit model.

**Functional Description :**
Clumpy is a land use and cover change model which aims to be used in sustainability and environmental sciences. It allows to observe past changes statistics, set future contrasted scenarios and allocate simulated future land use maps according to those projections.
From past land use maps and some explanatory variables, the software calibrates the model through machine learning methods (kernel density estimations, neighrest neighbors for now). Explicit probabilistic values are returned and can be then adjusted to contrasted scenarios set by the user. It is finally possible to allocate a simulated land use map in order to provide decision materials for public stakeholder.
Other land use and cover change model softwares such as Dinamica EGO, CLUE and LCM are compared to Clumpy in papers to come.

**Participants :**
François-Rémi Mazy,
Pierre-Yves Longaretti

**Contact :**
François-Rémi Mazy

## Install

*The software provided by pip is not up to date !*

build the package :
`python3 setup.py sdist bdist_wheel`

upload on testpypi
`python3 -m twine upload --repository testpypi dist/*`

upload on pypi
`twine upload dist/*`

help about pypi packages
https://packaging.python.org/tutorials/packaging-projects/

install package locally
python setup.py install

## JSON help

### available parameters

