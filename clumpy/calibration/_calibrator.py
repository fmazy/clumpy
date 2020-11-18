#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:48:12 2020

@author: frem
"""


class _Calibrator():
    def __init__(self, estimator, method, cv):
        self.method = method
        self.cv = cv