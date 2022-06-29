# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm

def determine_WoE_ranges(X,
                         params):
    
    n, d = X.shape
    
    ranges = []
    delta = []

    for k in range(d):
        print('d=',d)
        param = params[k]

        x = X[:, k].copy()
        n_round = _get_n_decimals(param['increment'])
        x = np.sort(x)
        x = np.round(x, n_round)

        ranges.append([np.round(x[0], n_round)])
        delta.append([0, 0])

        for xi in tqdm(x):
            if delta[k][-1] >= param['maximum_delta']:
                ranges[k].append(xi)
                delta[k].append(1)
            elif xi - ranges[k][-1] > param['increment'] and delta[k][-1] >= param['minimum_delta']:
                ranges[k].append(ranges[k][-1] + param['increment'])
                delta[k].append(1)

            elif len(ranges[k]) > 1:
                v1 = np.array([ranges[k][-1] - ranges[k][-2],
                                (delta[k][-2] - delta[k][-3])])
                v2 = np.array([xi - ranges[k][-1],
                                (delta[k][-1] + 1 - delta[k][-2])])

                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                if norm_v1 > 0 and norm_v2 > 0:
                    v1 /= norm_v1
                    v2 /= norm_v2

                    dot = v1[0] * v2[0] + v1[1] * v2[1]
                    if dot >= 0 and dot <= 1:
                        angle = np.arccos(np.abs(v1[0] * v2[0] + v1[1] * v2[1])) * 180 / np.pi
                    else:
                        angle = 0
                else:
                    angle = 0

                if angle > param['tolerance_angle'] and delta[k][-1] >= param['minimum_delta']:
                    ranges[k].append(xi)
                    delta[k].append(1)
                else:
                    delta[k][-1] += 1
            else:
                delta[k][-1] += 1

    return [np.array(r) for r in ranges]

def _get_n_decimals(s):
    try:
        int(s.rstrip('0').rstrip('.'))
        return 0
    except:
        return len(str(float(s)).split('.')[-1])