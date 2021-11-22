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
        The numpy array converted (it's a copy)

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

def smooth(x, window_len=3, window='hanning'):
    """
    function reference : https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    if window_len%2 == 0:
        return y[int(window_len / 2 - 1):-int(window_len / 2)]
    else:
        return y[int(window_len/2):-int(window_len/2)]


