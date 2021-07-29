# -*- coding: utf-8 -*-

import numpy as np

def human_size(s):
    """return size in a human readable form.

    Parameters
    ----------
    s : int
        size in octet.

    Returns
    -------
    s : int
        size in a human readable form

    """
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB']
    i = 0
    while round(s/1024) > 0:
        s = s / 1024
        i += 1
    
    return(s, units[i])

def ndarray_suitable_integer_type(a):
    """
    Convert a ndarray in a suitable integer type

    Parameters
    ----------
    a : ndarray
        numpy array to convert.

    Returns
    -------
    a_converted : ndarray
        The numpy array converted (it's a copy')

    """
    if a.min() >= 0: # if unsigned
        t = np.uint64
        m = a.max()
        if m <= 4294967295:
            t = np.uint32
            if m <= 65535:
                t = np.uint16
                if m<= 255:
                    t = np.uint8
    else:
        t = np.int64
        m = np.abs(a).max()
        if m <= 2147483647:
            t = np.uint32
            if m <= 32767:
                t = np.uint16
                if m<= 127:
                    t = np.uint8
                    
    return(a.astype(t))

def np_drop_duplicates_from_column(a, column_id):
    return(a[np.unique(a[:, column_id], return_index=True)[1]])
