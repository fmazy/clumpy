# -*- coding: utf-8 -*-

"""
randomMapping.py
====================================
The random mapping module of demeter
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import ndimage # used for distance computing
from matplotlib import pyplot as plt
import demeter.tools as dmtools
import demeter.definition
from PIL import Image
from matplotlib import pyplot as plt
import scipy.interpolate

class RandomCase(object):
    def __init__(self, size):
        self.size = size
        self.map_i = np.zeros(size)
        self.Z = []
        self.T = []
        
    def createUnifCase(self, N_v, N_Z, seed=None):
        if seed != None:
            np.random.seed(seed)
        self.map_i = np.random.randint(0, N_v, self.size)
        self.map_f = np.random.randint(0, N_v, self.size)
        
        for k in range(N_Z):
            self.Z.append(np.random.random(self.size))
            
    def createIntCase(self, N_v, N_q, seed=None):
        if seed != None:
            np.random.seed(seed)
        self.map_i = np.random.randint(0, N_v, self.size)
        self.map_f = np.random.randint(0, N_v, self.size)
        
        for k in range(len(N_q)):
            self.Z.append(np.random.randint(0, N_q[k], self.size))
                                    
    def createZ(self, samples = 10, interp='thin_plate', seed=None):     
        #interp_types = ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate']
        if seed != None:
            np.random.seed(seed)
        x = np.random.randint(0,self.size[0],samples)
        y = np.random.randint(0,self.size[1],samples)
        z = np.random.random(samples)
        
        interp = scipy.interpolate.Rbf(x, y, z, function=interp)
        
        xi, yi = np.mgrid[0:self.size[0], 0:self.size[1]]
        
        Zk = dmtools.normalize(interp(xi, yi))
        
        self.Z.append(Zk)
#        return(Zk)
        
    def createMap_i(self, P_v = [0.6,0.25,0.15], samples=10, interp='thin_plate', seed=None):
        if seed != None:
            np.random.seed(seed)
        x = np.random.randint(0,self.size[0],samples)
        y = np.random.randint(0,self.size[1],samples)
        z = np.random.random(samples)
        
        interp = scipy.interpolate.Rbf(x, y, z, function=interp)
        
        xi, yi = np.mgrid[0:self.size[0], 0:self.size[1]]
        
        self.v = dmtools.normalize(interp(xi, yi), low=0, high=0.99)
        
        self.map_i = np.digitize(self.v, bins=np.cumsum(P_v))
        
    def addTransition(self, vi, vf, P, z_mean, z_sd):
            t = Transition(vi, vf, P, z_mean, z_sd)
            self.T.append(t)
            
    def createMap_f(self):
        self.map_f = self.map_i.copy()
        for T in self.T:
            print("from "+str(T.vi)+" to "+str(T.vf))
            J_vi = np.argwhere(self.map_i.flat == T.vi)
            for j in tqdm(J_vi):
                P = T.P
                for k, Zk in enumerate(self.Z):
                    P = P*normal_func(Zk.flat[j], T.z_mean[k], T.z_sd[k])
                if np.random.rand() <= P:
                    self.map_f.flat[j] = T.vf
            
    
    def displayMap_i(self):
        plt.imshow(self.map_i)
        plt.colorbar()
        plt.show()
        
    def displayMap_f(self):
        plt.imshow(self.map_f)
        plt.colorbar()
        plt.show()
        
    def displayZk(self, i):
        plt.imshow(self.Z[i], cmap='gist_earth')
        plt.colorbar()
        plt.show()
        
    def exportCase(self, folder_path):       
        img_map_i = Image.fromarray(self.map_i.astype('uint8'))
        img_map_i.save(folder_path+'map_i.tif')
        
        img_map_f = Image.fromarray(self.map_f.astype('uint8'))
        img_map_f.save(folder_path+'map_f.tif')
        
        for k,Zk in enumerate(self.Z):
            img_Z = Image.fromarray(Zk)
            img_Z.save(folder_path+'Z'+str(k)+'.tif')
        
        

class Transition(object):
    def __init__(self, vi, vf, P, z_mean, z_sd):
        self.vi = vi
        self.vf = vf
        self.P = P
        self.z_mean = z_mean
        self.z_sd = z_sd
        
def normal_func(x, mu, sigma):
    y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )
    return(y)