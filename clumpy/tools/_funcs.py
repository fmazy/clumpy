# -*- coding: utf-8 -*-
import inspect

def extract_parameters(func, kwargs):
    sig = inspect.signature(func)
    parameters = {}
    func_parameters = list(sig.parameters.keys())

    for key, value in kwargs.items():
        if key in func_parameters:
            parameters[key] = value

    return (parameters)