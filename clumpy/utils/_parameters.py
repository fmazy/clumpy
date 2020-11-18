#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:15:58 2020

@author: frem
"""

import numpy as np

def check_list_parameters_vi(list_parameters_vi, list_vi = None):
    """
    Raise an error if the list of parameters function of vi is uncorrect.

    Parameters
    ----------
    list_parameters_vi : list(dict)
        List of parameters function of vi.
    list_vi : list, default=None
        List of initial states vi. If None, the test is passed.
    """
    
    if not isinstance(list_parameters_vi, list):
        raise(TypeError("list_parameters_vi should be a list."))
    
    first_keys = list_parameters_vi[0].keys()
    
    if list_vi is not None:
        if len(first_keys) != len(list_vi):
            raise(ValueError("The parameters keys are wrong. They should be "+str(list_vi)))
        if not all(np.sort(list(first_keys)) == np.sort(list_vi)):
            raise(ValueError("The parameters keys are wrong. They should be "+str(list_vi)))
    
    for parameter_vi in list_parameters_vi:
        check_parameter_vi(parameter_vi)
        if first_keys != parameter_vi.keys():
            raise(ValueError("The parameters keys, i.e. vi should be the same."))


def check_parameter_vi(parameter_vi):
    """
    Raise an error if the parameter function of vi is uncorrect.

    Parameters
    ----------
    parameter_vi : dict
        The parameter function of vi.
        
    error_output : bool, default=True
        Raise an error if True. Else, return False.

    Returns
    -------
    check_result : Boolean
        `True` if the parameter passed the test.
    """
    
    if not isinstance(parameter_vi, dict):
        raise(TypeError("parameter_vi should be a dict with vi as keys."))
    
    if not all(isinstance(k, int) for k in parameter_vi.keys()):
        raise(ValueError("All parameter_vi keys should be int. Get "+str(parameter_vi.keys())))
        
def fill_default_parameter_vi(parameter_vi, default):
    """
    Fill the parameter function of vi with default if ``'default'`` for each vi. The parameter is considered as valid. Please first use the function :function:`check_parameter_vi` or :function:`check_list_parameters_vi`.

    Parameters
    ----------
    parameter_vi : dict
        The parameter function of vi.
    default : *
        The default value.

    Returns
    -------
    The filled parameter.

    """
    
    for vi in parameter_vi.keys():
        if parameter_vi[vi] == 'default':
            parameter_vi[vi] = default
        
    return(parameter_vi)