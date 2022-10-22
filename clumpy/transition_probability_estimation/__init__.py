#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transition probability estimation methods
"""

from ._bayes import Bayes
from ._bayes_ekde import BayesEKDE
from ._importer import Importer

_methods = {'bayes' : Bayes,
            'bayes_ekde' : BayesEKDE,
            'importer' : Importer}