# Clumpy

Comprehensive Land Use Models in Python

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

## to do

- expansion de taches
- passage des algorithmes d'allocation en générique : le sampling est l'affaire de la méthode mais le reste est commun...
- feature selection avec RFE et RFECV
    - feature selection n'est pas si simple car on doit recalculer la cible. En effet, la cible dépend du nombre de features.
- SVM calibration
- neural networks calibration
- avoir un protocol correct de calibration qui inclut 
    - normalisation de P_vf__vi_z pour avoir des valeurs significatives que l'on retrouve ensuite avec un inverse_transform
    - feature selection
    - feature independance test