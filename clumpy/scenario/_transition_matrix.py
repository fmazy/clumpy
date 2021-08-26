#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 10:47:48 2021

@author: frem
"""

import numpy as np

class TransitionMatrix():
    def __init__(self, M, list_u=None, list_v=None):
        
        if list_u is None or list_v is None:
            M, list_u, list_v = _compact_transition_matrix(full_M = M)
        
        M = np.nan_to_num(M)
        
        # check
        if not np.all(np.isclose(np.sum(M, axis=1), np.ones(M.shape[0]))):
            print("Warning : The transition matrix is uncorrect. The rows should sum to one")
            
        
        self.M = M
        self.list_u = list(list_u)
        self.list_v = list(list_v)
    
    def get_value(self, u, v):
        return(self.M[self.list_u.index(u), self.list_v.index(v)])
    
    def copy(self):
        return(TransitionMatrix(self.M.copy(), self.list_u, self.list_v))
    
    def set_value(self, a, u, v):
        self.M[self.list_u.index(u), self.list_v.index(v)] = a
    
    
    def select_u(self, u):
        M = self.M[[self.list_u.index(u)],:].copy()
        
        # list_v = []
        # for id_v, v in enumerate(self.list_v):
            # if M[0, id_v] > 0:
                # list_v.append(v)
        
        return(TransitionMatrix(M, [u], self.list_v))
    
    def P_v(self, u):
        p = self.M[self.list_u.index(u), :].copy()
        
        return(p, self.list_v)
    
    def __repr__(self):
        return('TransitionMatrix()')
    
    def full_matrix(self):
        max_v = np.max(self.list_v)
        
        full_M = np.diag(np.ones(max_v+1))
    
        list_v_full_matrix = list(np.arange(max_v+1))
    
        for id_u, u in enumerate(self.list_u):
            for id_v, v in enumerate(self.list_v):
                full_M[list_v_full_matrix.index(u),
                       list_v_full_matrix.index(v)] = self.M[id_u, id_v]
        
        full_M = full_M.astype(float)
        
        full_M = np.nan_to_num(full_M)
        
        return(full_M)
        
    def multisteps(self, n):
        tp = self.full_matrix()
        
        eigen_values, P = np.linalg.eig(tp)
        
        eigen_values = np.power(eigen_values, 1/n)
        
        full_M = np.dot(np.dot(P, np.diag(eigen_values)), np.linalg.inv(P))
        
        compact_M, _, _ = _compact_transition_matrix(full_M = full_M,
                                               list_u = self.list_u,
                                               list_v = self.list_v)
        
        return(TransitionMatrix(compact_M, self.list_u, self.list_v))
    
    def patches(self, patches):
        M_p = np.zeros_like(self.M)
        
        for id_u, u in enumerate(self.list_u):
            M_p[id_u, self.list_v.index(u)] = 1.0
            for id_v, v in enumerate(self.list_v):
                if u != v:
                    if patches[u][v]['area'].size > 0 and self.M[id_u, id_v] > 0:
                        M_p[id_u, id_v] = self.M[id_u, id_v] / patches[u][v]['area'].mean()
                        M_p[id_u, self.list_v.index(u)] -= M_p[id_u, id_v]
        
        return(TransitionMatrix(M_p, self.list_u, self.list_v))

def compute_transition_matrix(V_u, list_u, list_v):
    M = np.zeros((len(list_u), len(list_v)))
    
    for id_u, u in enumerate(list_u):
        for id_v, v in enumerate(list_v):
            M[id_u, id_v] = np.sum(V_u[u] == v)
    
    return(TransitionMatrix(M, list_u, list_v))

def load_transition_matrix(path):
    data = np.genfromtxt(path, delimiter=',')
    
    list_u = list(data[1:,0].astype(int))
    list_v = list(data[0,1:].astype(int))
    
    M = data[1:,1:]
    
    return(TransitionMatrix(M, list_u, list_v))

def _compact_transition_matrix(full_M, list_u=None, list_v=None):
    if list_u is None or list_v is None:
        list_u = []
        list_v = []
        
        for u in range(full_M.shape[0]):
            v_not_null = np.arange(full_M.shape[1])[full_M[u, :] > 0]
            
            # if transitions (u -> u is not a transition)
            if len(v_not_null) > 1:
                if u not in list_u:
                        list_u.append(u)
                for v in v_not_null:
                    if v not in list_v:
                        list_v.append(v)
        
        list_u.sort()
        list_v.sort()
        
    list_v_full_matrix = list(np.arange(full_M.shape[0]))
        
    compact_M = np.zeros((len(list_u), len(list_v)))
    
    for id_u, u in enumerate(list_u):
        for id_v, v in enumerate(list_v):
            compact_M[id_u, id_v] = full_M[list_v_full_matrix.index(u),
                                           list_v_full_matrix.index(v)]
    
    return(compact_M, list_u, list_v)

