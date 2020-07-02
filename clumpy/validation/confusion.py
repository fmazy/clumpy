"""
intro
"""

import numpy as np
import pandas as pd
from scipy import ndimage


from .. import definition

def confusion_matrix(map_i, map_actual, map_simulated, T):
    
    print('J creation')
    J = definition.data.create_J(map_i, map_actual, T)
    definition.data.restrict_vf_to_T(J, T, inplace=True)
    
    J.rename(columns={'vf':'vf_actual'}, inplace=True)
    J['vf_simulated'] = map_simulated.data.flat[J.index.values]
    
    print('done')
    
    for Ti in T.Ti.values():
        for Tif in Ti.Tif.values():
            print(Tif)
            
            N_good_prediction = J.loc[(J.vi==Ti.vi) &
                                    (J.vf_actual == Tif.vf) &
                                    (J.vf_simulated == Tif.vf)].index.size
            
            N_total = J.loc[(J.vi==Ti.vi) &
                            (J.vf_actual == Tif.vf)].index.size
            
            print('total : '+str(N_total)+' good : '+str(N_good_prediction)+' - '+str(N_good_prediction/N_total))
            
    

    # confusion_matrix = pd.DataFrame(columns=['vi', 'vf', 'N', 'N_predicted'])
    
    return(J)

def fuzzy_kappa_simulation(map_i, map_actual, map_simulated, T, sigma):
    """
    from Van Vliet et al (2013)
    
    T is used to focus validation on specified transitions
    """
    
    
    print('J creation')
    J = definition.data.create_J(map_i, map_actual, T)
    definition.data.restrict_vf_to_T(J, T, inplace=True)
    
    J.rename(columns={'vf':'vf_actual'}, inplace=True)
    J['vf_simulated'] = map_simulated.data.flat[J.index.values]
    
    print('done')
    
    print('transition distance computing')
    
    delta_vi_vf_names = []
    
    for Ti in T.Ti.values():
        for Tif in Ti.Tif.values():            
            print(Tif)
            
            for suffix in ['actual', 'simulated']:
                print('\t'+suffix)
                J_delta = J.copy()
                for Ti_delta in T.Ti.values():
                    for Tif_delta in Ti_delta.Tif.values():
                        print('\t\t '+str(Tif_delta))

                        M_vi_vf = np.zeros(map_i.shape)
                        
                        M_vi_vf.flat[J_delta.loc[(J_delta.vi == Ti_delta.vi) &
                                                 (J_delta['vf_'+suffix] == Tif_delta.vf)].index.values] = 1
                        
                        distance = ndimage.distance_transform_edt(1 - M_vi_vf)
                        
                        delta_vi_vf_name = 'delta_vi'+str(Ti_delta.vi)+'_vf'+str(Tif_delta.vf)
                        delta_vi_vf_names.append(delta_vi_vf_name)
                        
                        J_delta[delta_vi_vf_name] = distance.flat[J_delta.index.values] + 1 # minimal distance is 1 (for distance decay function)
                        
                        # f function is f'x)=1/x
                        J_delta[delta_vi_vf_name] = sigma[Ti.vi, Tif.vf, Ti_delta.vi, Tif_delta.vf]/J_delta[delta_vi_vf_name]
                
                J['delta_'+suffix+'_vi'+str(Ti.vi)+'_vf'+str(Tif.vf)] = J_delta[delta_vi_vf_names].max(axis=1)
    
    J['Delta_a'] = -1
    J['Delta_s'] = -1
    
    for Ti_simulated in T.Ti.values():
        for Tif_simulated in Ti_simulated.Tif.values():
            print(Tif_simulated)
            
            for Ti_actual in T.Ti.values():
                for Tif_actual in Ti_actual.Tif.values():
                    print('\t '+str(Tif_actual))
                    
                    index_s = J.loc[(J.vi==Ti_simulated.vi) &
                                    (J.vf_simulated == Tif_simulated.vf)].index
                    
                    index_a = J.loc[(J.vi==Ti_actual.vi) &
                                    (J.vf_actual == Tif_actual.vf)].index
                    
                    J.loc[index_s,'Delta_a'] = J.loc[index_s, 'delta_actual_vi'+str(Ti_simulated.vi)+'_vf'+str(Tif_simulated.vf)]
                    
                    J.loc[index_a,'Delta_s'] = J.loc[index_a, 'delta_simulated_vi'+str(Ti_actual.vi)+'_vf'+str(Tif_actual.vf)]
    
    J['Delta'] = J[['Delta_a', 'Delta_s']].min(axis=1)
    J.drop(['Delta_a', 'Delta_s'], axis=1, inplace=True)
    
    return(J)
            
            
            