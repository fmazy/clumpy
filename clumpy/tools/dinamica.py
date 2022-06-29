# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
# from tqdm import tqdm

from ._path import create_directories, path_split

def determine_WoE_ranges(X,
                         params):
    
    n, d = X.shape
    
    ranges = []

    for k in range(d):
        param = params[k]

        x = np.array(X[:, k])
        n_round = _get_n_decimals(param['increment'])
        x = np.sort(x)
        # x = np.round(x, n_round)

        r = [np.round(x[0], n_round)]
        delta = [0, 0]
        
        # print(r, x[1])

        for xi in x:
            # print(xi - r[-1], param['increment'], delta[-1], param['minimum_delta'])
            if delta[-1] >= param['maximum_delta']:
                r.append(xi)
                delta.append(1)
            elif xi - r[-1] > param['increment'] and delta[-1] >= param['minimum_delta']:
                r.append(np.round(xi, n_round)+10**-n_round)
                delta.append(0)
                # break
            
            
            # elif delta[-1] > param['minimum_delta']:
            #     v1 = np.array([r[-1] - r[-2],
            #                     (delta[-2] - delta[-3])])
            #     v2 = np.array([xi - r[-1],
            #                     (delta[-1] + 1 - delta[-2])])
                
            #     # print('v1',v1)
            #     # print('v2',v2)

            #     norm_v1 = np.linalg.norm(v1)
            #     norm_v2 = np.linalg.norm(v2)
            #     if norm_v1 > 0 and norm_v2 > 0:
            #         v1 /= norm_v1
            #         v2 /= norm_v2

            #         dot = v1[0] * v2[0] + v1[1] * v2[1]
            #         if dot >= 0 and dot <= 1:
            #             angle = np.arccos(np.abs(v1[0] * v2[0] + v1[1] * v2[1])) * 180 / np.pi
            #         else:
            #             angle = 0
            #     else:
            #         angle = 0
                
            #     # print('angle', angle, xi - r[-1], delta[-1])
                
            #     if angle > param['tolerance_angle']:
            #         # print('!')    
            #         r.append(np.round(xi, n_round)+10**-n_round)
            #         delta.append(0)
            #         # break
            #     else:
            #         delta[-1] += 1
            
            else:
                delta[-1] += 1
            #     # break
        
        # r.append(np.round(xi, n_round)+10**-n_round)
        ranges.append(r)
            

    return [np.array(r) for r in ranges]

def save_WoE_ranges(ranges, names, path):
    d = len(ranges)
    columns = ['From*', 'To*', 'Variable*', 'Range_Lower_Limit*', 'Weight']

    df_all = pd.DataFrame(columns=columns)
    
    for k in range(d):
        df = pd.DataFrame(columns=columns)
        df['Range_Lower_Limit*'] = ranges[k]
        df['From*'] = 1
        df['To*'] = 2
        df['Variable*'] = names[k]
        df['Weight'] = 0
        df_all = pd.concat((df_all, df))
    
    folder_path, file_name, ext = path_split(path)
    create_directories(folder_path)
    df_all.to_csv(path, index=False)

def _get_n_decimals(s):
    try:
        int(s.rstrip('0').rstrip('.'))
        return 0
    except:
        return len(str(float(s)).split('.')[-1])