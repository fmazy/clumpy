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

## Dev

### GUI et JSON

On est pas forcément obligé de partir de QGis...

Le cas est entièrement décrit par un dictionnaire case.params. Celui-ci peut être exporté / chargé par un fichier json qui le reprend exactement. L'interface graphique permet de modifier directement ce dictionnaire case.params. On peut le modifier sans rien prendre en compte. Le cas ne sera concrètement construit que lorsque l'on appelle la fonction case.make(). La fonction case.run() réalise un fit() puis la simulation selon les paramètres entrés.

La fenêtre est séparée en deux colonnes.
L'idée est d'avoir un arbre (tree) à gauche qui suit cette architecture :

- Case
    -Regions
        - region_A (nom de la région)
            - Features
                - x0 (nom de la feature)
                - x1 (nom de la feature)
                - x2 (nom de la feature)
                - distance to 2
            - Calibration
            - Allocation
        - region_B (nom de la région)
            - Features
                - x0 (nom de la feature)
                - x1 (nom de la feature)
                - x2 (nom de la feature)
                - distance to 2
            - Calibration
            - Allocation

La colonne de droite change selon ce que l'on clique dans l'arbre à gauche.

Je liste ci-dessous les éléments à afficher si on clique sur tel ou tel élément.

- Case : Ici, on affiche les paramètres généraux du cas. Il y a également le bouton pour lancer le cas.
    - open : bouton select file .json puis appelle case.open()
    - save : appelle case.save()
    - save as : bouton new file .json puis appelle case.save_as(new_file)
    - Output folder : champ select folder
    - verbose : champ integer >= 0
    - palette : champ select file .qml
    - lul initial : champ select file .tif
    - lul final : champ select file .tif
    - computes transition probabilities only : check box (boolean)
    - Run : bouton qui appelle case.make() et case.run()
- Regions : 
    - New region : champ ligne, bouton créer, appelle case.new_region(label)
- Nom d'une région
    - transition matrix : champ select file .csv
    - calibration mask : champ select file .tif, peut-être vide.
    - allocation mask : champ select file .tif, peut-être vide.
    - 
