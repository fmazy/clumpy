"""
intro
"""

from .. import definition
from ..definition import transition

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from .. import tools

def local_dependence_derivative(J):
    J = J[['v','z']].copy()
    N_vi_vf_z = J.groupby(J.columns.to_list()).size().reset_index(name=('N_vi_vf_z',''))
    print(N_vi_vf_z)
    return(True)

def compute_P_vf__vi(J, name='P_vf__vi'):
    
    # P_vf__vi = J.groupby([('v','i'), ('v','f')]).size().reset_index(name=('N_vi_vf',''))
    
    # N_vi = J.groupby([('v','i')]).size().reset_index(name=('N_vi',''))
    
    # P_vf__vi = P_vf__vi.merge(N_vi, how='left')
    
    # return(P_vf__vi)
    
    # for vf in P_vf__vi.v.f.unique():
    #     P_vf__vi[name, vf] = P_vf__vi
    
    P_vf__vi = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('v','i')]))
    
    df = J.groupby([('v','i'), ('v','f')]).size().reset_index(name=('N_vi_vf',''))
    
    for vi in df.v.i.unique():
        df.loc[df.v.i==vi, ('P_vf__vi','')] = df.loc[df.v.i==vi, 'N_vi_vf']/ df.loc[df.v.i==vi, 'N_vi_vf'].sum()
        
        P_vf__vi.loc[P_vf__vi.index.size, ('v','i')] = vi
        for vf in df.loc[df.v.i==vi].v.f.unique():
            P_vf__vi.loc[P_vf__vi.v.i==vi,('P_vf__vi', vf)] = df.loc[(df.v.i==vi) & (df.v.f==vf)].P_vf__vi.values[0]
    
    return(P_vf__vi)
    


def compute_P_zk__vi(J, T, alpha):
    df = pd.DataFrame(columns=['vi','Zk_name','q','P_zk__vi'])
    
    for Ti in T.Ti.values():
        # restriction to considered pixels
        J_vi = J.loc[(J.v.i==Ti.vi)]
        for Zk in Ti.Z.values():
            
            # restriction to considered alpha
            alpha_Zk = alpha.loc[(alpha.vi == Ti.vi) &
                                 (alpha.Zk_name == Zk.name)]
            
            # we count every unique combinaisons of vi, Zk
            count = J_vi.z.groupby([Zk.name]).size().reset_index(name='P_zk__vi')
            
            # we fill holes where no occurences have been found
            q = count[Zk.name].values
            n = count['P_zk__vi'].values
            n_full = np.zeros((alpha_Zk.index.size+1))
            n_full[q.astype(int)] = n
            
            # sub df creation
            df_sub = pd.DataFrame(columns=['vi','Zk_name','q','P_zk__vi'])
            df_sub.q = np.arange((alpha_Zk.index.size+1))
            df_sub.P_zk__vi = n_full/n_full.sum()
            df_sub['vi'] = Ti.vi
            df_sub['Zk_name'] = Zk.name
            
            # df concatenation
            df = pd.concat([df, df_sub], ignore_index=True)
    
    return(df)

def compute_P_zk__vi_vf(J, T, alpha):
    
    # cols = ['v','i', ]
    
    df = pd.DataFrame(columns=['vi','vf','Zk_name','q','P_zk__vi_vf'])
    
    for Ti in T.Ti.values():
        for Zk in Ti.Z.values():
            for Tif in Ti.Tif.values():
                # restriction to considered pixels
                J_vi_vf = J.loc[(J.v.i==Ti.vi) & (J.v.f==Tif.vf)]
                
                    
                # restriction to considered alpha
                alpha_Zk = alpha.loc[(alpha.vi == Ti.vi) &
                                     (alpha.Zk_name == Zk.name)]
                
                # we count every unique combinaisons of vi, vf, Zk
                count = J_vi_vf.z.groupby([Zk.name]).size().reset_index(name='P_zk__vi_vf')
                
                # we fill holes where no occurences have been found
                q = count[Zk.name].values
                n = count['P_zk__vi_vf'].values
                n_full = np.zeros((alpha_Zk.index.size+1))
                n_full[q.astype(int)] = n
                
                # sub df creation
                df_sub = pd.DataFrame(columns=['vi','vf','Zk_name','q','P_zk__vi_vf'])
                df_sub.q = np.arange((alpha_Zk.index.size+1))
                df_sub.P_zk__vi_vf = n_full / n_full.sum()
                df_sub['vi'] = Ti.vi
                df_sub['vf'] = Tif.vf
                df_sub['Zk_name'] = Zk.name
                
                # df concatenation
                df = pd.concat([df, df_sub], ignore_index=True)
    
    return(df)

def plot_P_zk__vi(P_zk__vi, vi, Zk_name, max_one=False, sum_one=False, color=None, linestyle='-', linewidth=1.5, label=None, alpha=None, step=True):
    """
    plots a given :math:`P_k(\hat{z}|v_i,v_f)`.
    
    """
    # for colors, see https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    
    P_zk__vi = P_zk__vi.loc[(P_zk__vi.vi == vi) &
                          (P_zk__vi.Zk_name == Zk_name)]
    
    y = P_zk__vi.P_zk__vi.values
    y = y[1:-1]
    
    if type(alpha) == pd.DataFrame:
        x = alpha.loc[(alpha.vi == vi) &
                      (alpha.Zk_name == Zk_name)].alpha.values
        x = x[0:-1]
    else:
        x = P_zk__vi.q.values
        x = x[1:-1]
    
    if sum_one:
        y = y/np.sum(y)
        
    if max_one:
        y = y/np.max(y)
    
    if step:
        plt.step(x=x,
                y=y,
                where='post',
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label)
    else:
        plt.plot(x,
                y,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label)
    
def plot_P_zk__vi_vf(P_zk__vi_vf, vi, vf, Zk_name, max_one=False, sum_one=False, color=None, linestyle='-', linewidth=1.5, label=None, alpha=None, step=True):
    """
    plots a given :math:`P_k(\hat{z}|v_i,v_f)`.
    

    """
    # for colors, see https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    
    P_zk__vi_vf = P_zk__vi_vf.loc[(P_zk__vi_vf.vi == vi) &
                                (P_zk__vi_vf.vf == vf) &
                                (P_zk__vi_vf.Zk_name == Zk_name)]
    
    
    y = P_zk__vi_vf.P_zk__vi_vf.values
    y = y[1:-1]
    
    if type(alpha) == pd.DataFrame:
        x = alpha.loc[(alpha.vi == vi) &
                      (alpha.Zk_name == Zk_name)].alpha.values
        x = x[0:-1]
    else:
        x = P_zk__vi_vf.q.values
        x = x[1:-1]
        
    if sum_one:
        y = y/np.sum(y)
        
    if max_one:
        y = y/np.max(y)
    
    if step:
        plt.step(x=x,
                y=y,
                where='post',
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label)
    else:
        plt.plot(x,
                y,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label)

def export(T:transition.Transition,
                      folder_path,
                      Z_alpha_trigger = True,
                      Z_infos_trigger = True,
                      N_zk_vi_trigger = False,
                      N_zk_vi_vf_trigger = True,
                      P_vf__vi_trigger = True,
                      patchesHist_trigger = True):
    """
    export the transition case.
    
    :param T: transition object
    :type T: :class:`.definition.transition.Transition`
    :param folder_path: path to export fodler
    :type folder_path: string
    :param Z_alpha_trigger: if True, export Z_alpha, i.e. bins of :math:`Z_k`
    :type Z_alpha_trigger: bool, default=True
    :param Z_infos_trigger: if True, export Z_infos which details :math:`Z_k` informations such as name or index.
    :type Z_infos_trigger: bool, default=True
    :param N_zk_vi_trigger: if True, export :math:`P_k(\hat{z}|v_i)`.
    :type N_zk_vi_trigger: bool, default=True
    :param N_zk_vi_vf_trigger: if True, export :math:`P_k(\hat{z}|v_i,v_f)`.
    :type N_zk_vi_vf_trigger: bool, default=True
    :param P_vf__vi_trigger: if True, export :math:`P_k(v_f|v_i)`.
    :type P_vf__vi_trigger: bool, default=True
    :param patchesHist_trigger: if True, export patches histogram.
    :type patchesHist_trigger: bool, default=True
    """
    
    df_Z_infos = pd.DataFrame(columns = ['vi', 'k', 'discrete', 'dyn'])
    
    for Ti in T.Ti.values():            
        for k, Zk in Ti.Z.items(): 
            df_Z_infos.loc[df_Z_infos.k.size] = [Ti.vi,
                                        Zk.k,
                                        Zk.discrete,
                                        Zk.dyn]
        
    print('csv writing')
    if Z_alpha_trigger:
        T.Z_alpha.to_csv(folder_path+'Z_alpha.csv', index=False)
    if Z_infos_trigger:
        df_Z_infos.to_csv(folder_path+'Z_infos.csv', index=False)
    if N_zk_vi_trigger:
        T.N_zk_vi.to_csv(folder_path+'N_zk_vi.csv', index=False)
    if N_zk_vi_vf_trigger:
        T.N_zk_vi_vf.to_csv(folder_path+'N_zk_vi_vf.csv', index=False)
    if P_vf__vi_trigger:
        T.P_vf__vi.to_csv(folder_path+'P_vf__vi.csv', index=False)
    if patchesHist_trigger:
        T.patchesHist.to_csv(folder_path+'patchesHist.csv', index=False)
    print('done')
    
def load(T:transition.Transition,
                      folder_path,
                      Z_alpha_trigger = True,
                      Z_infos_trigger = True,
                      N_zk_vi_trigger = True,
                      N_zk_vi_vf_trigger = True,
                      P_vf__vi_trigger = True,
                      patchesHist_trigger = True):
    """
    import a transition case.
    
    :param T: transition object
    :type T: :class:`.definition.transition.Transition`
    :param folder_path: path to import folder
    :type folder_path: string
    :param Z_alpha_trigger: if True, import Z_alpha, i.e. bins of :math:`Z_k`
    :type Z_alpha_trigger: bool, default=True
    :param Z_infos_trigger: if True, import Z_infos which details :math:`Z_k` informations such as name or index.
    :type Z_infos_trigger: bool, default=True
    :param N_zk_vi_trigger: if True, import :math:`P_k(\hat{z}|v_i)`.
    :type N_zk_vi_trigger: bool, default=True
    :param N_zk_vi_vf_trigger: if True, import :math:`P_k(\hat{z}|v_i,v_f)`.
    :type N_zk_vi_vf_trigger: bool, default=True
    :param P_vf__vi_trigger: if True, import :math:`P_k(v_f|v_i)`.
    :type P_vf__vi_trigger: bool, default=True
    :param patchesHist_trigger: if True, import patches histogram.
    :type patchesHist_trigger: bool, default=True
    """
    
    # les bins d'abord
    
    if P_vf__vi_trigger:
        T.P_vf__vi = pd.read_csv(folder_path+'P_vf__vi.csv')
    if N_zk_vi_trigger:
        T.N_zk_vi= pd.read_csv(folder_path+'N_zk_vi.csv')
    if N_zk_vi_vf_trigger:
        T.N_zk_vi_vf = pd.read_csv(folder_path+'N_zk_vi_vf.csv')
    if patchesHist_trigger:
        T.patchesHist = pd.read_csv(folder_path+'patchesHist.csv')
    
    if Z_alpha_trigger:
        T.Z_alpha = pd.read_csv(folder_path+'Z_alpha.csv')
    if Z_infos_trigger:
        # ce truc ne sert à rien...
        df_Z = pd.read_csv(folder_path+'Z_infos.csv')

    print("done")


        

    