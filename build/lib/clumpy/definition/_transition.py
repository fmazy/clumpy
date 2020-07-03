class _Transition(object):
    """
    Base object of a case, such as for a calibration or an allocation stage.
    
    Parameters
    ==========
    map_i : LandUseCoverLayer
        initial LUC layer
    map_f : LandUseCoverLayer (default=``None``)
        final LUC layer, only required for the calibration stage     
    """
    def __init__(self):
        self.Ti = {}

    def addTi(self, vi):
        """
        adds a transition structure for an initial state.
        
        :param vi: initial state
        :type vi: int
        
        :returns: a :class:`._Transition_vi` object.
        """
        if vi not in self.Ti.keys():
            self.Ti[vi] = _Transition_vi(vi, self)
        return(self.Ti[vi])
    
    def addTif(self, vi, vf):
        """
        adds a transition structure for an initial state and a different final state.
        
        :param vi: initial state
        :type vi: int
        :param vf: final state
        :type vf: int
        """
        Ti = self.addTi(vi)
        Tif = Ti.addTif(vf)
        
        return(Tif)
    
    def get_all_possible_vf(self):
        list_vf = []
        for Ti in self.Ti.values():
            for vf in Ti.Tif.keys():
                if vf not in list_vf:
                    list_vf.append(vf)
        return(list_vf)
    
    def get_transition_tuples(self):
        l = []
        for Ti in self.Ti.values():
            for vf in Ti.Tif.keys():
                l.append((Ti.vi, vf))
        return(l)
    
            
    def __str__(self):
        txt = ""
        for Ti in self.Ti:
            for Tif in Ti.Tif:
                txt += str(Tif)+"\n"
        txt = txt[:-1]
        return(txt)

            
class _Transition_vi(object):
    """
    Transition with :math:`v_i` fixed.
    
    :param vi: initial state
    :type vi: int
    :param transition: transition object
    :type transition: :class:.`_Transition`
    """
    def __init__(self, vi, transition):
        self.vi = int(vi)
        self.T = transition
        self.Tif = {}
        self.Z = {}
                
        # J_vi = np.argwhere(self.T.map_i.data.flat == self.vi).transpose()[0]
        # J_vi correspond aux indices dans J soit les véritables id des pixels
        # self.J_vi = pd.DataFrame()
        # self.J_vi['j'] = J_vi
        # the index of the dataframe corresponds to j
        # self.J_vi.set_index('j', drop=True, inplace=True)
        
        # if self.T.map_f != None:
            # self.J_vi['vf'] = self.vi
        
        # self.P = pd.DataFrame()
        
    def addTif(self, vf):
        """
        adds a transition structure for an initial state and a different final state.
        
        :param vf: final state
        :type vf: int
        """
        # this vf already exists ?
        if vf in self.Tif.keys():
            print("WARNING: this Tif does already exist !")
        else:
            self.Tif[vf] = _Transition_vi_vf(vf, self)
        return(self.Tif[vf])
    
    # def getNewK(self):
    #     # on récupère tous les indices déjà en place.
    #     # si il y a un trou on retourne l'indice correspondant
    #     new_k = 0
    #     for Zk in self.Z:
    #         if new_k < Zk.k:
    #             return(new_k)
    #         else:
    #             new_k += 1
    #     return(new_k)
        
    def __str__(self):
        txt = "transition vi="+str(self.vi)+"\n"
        txt += "explanatory variables:\n"
        for Zk in self.Z:
            txt += "\t"+Zk.name+"\n"
            txt += "\t\t[min, max]=["+str(round(Zk.min,2))+","+str(round(Zk.max,2))+"]\n"
        txt = txt[:-1]
        return(txt)
        
class _Transition_vi_vf(object):
    """
    Transition with :math:`v_i` and :math:`v_f` fixed.
    
    :param vf: final state
    :type vf: int
    :param Ti: transition with :math:`v_i` fixed object
    :type Ti: :class:`._Transition_vi`
    """
    def __init__(self, vf, Ti):
        self.vi = Ti.vi
        self.vf = int(vf)
        self.Ti = Ti
        # self.J_vi_vf = None
        # self.patches_param = {'isl': {'S_max':math.inf,
                                      # 'bins':'auto'},
                              # 'exp': {'S_max':math.inf,
                                      # 'bins':'auto'}}
        
        # if self.Ti.T.map_f is not None:
            # J_vf = np.argwhere(self.Ti.T.map_f.data.flat == self.vf).transpose()[0]
            # df = pd.DataFrame()
            # df['j'] = J_vf
            # df['vf'] = self.vf
            # df.set_index('j', drop=True, inplace=True)
                        
            # self.Ti.J_vi.update(other=df,
                                  # join='left',
                                  # overwrite=True)
                    
#    def triggerEV(self, *args):
#        """
#        selected explanatory variables are triggered as effective EV for this transition
#        
#        :param *args: explanatory variable :math:`Z_k`
#        :type *args: :class:`.explanatory_variable.Zk`
#        """
#        for Zk in args:
#            if Zk not in self.Ti.Z.values():
#                print(Zk.k+' not defined for vi='+str(self.Ti.vi))
#                return("ERROR")
#            self.Z[Zk.k] = Zk
#    
#    def addTriggerEv(self, Zk):
#        self.Z[Zk.k] = Zk
#        
#    def __str__(self):
#        return("T"+str(self.vi)+"->"+str(self.vf))
#        
#    def addLayerP_vf__vi_z(self, data=None, path=None):
#        self.layer_P_vf__vi_z = layer.LayerP_vf__vi_z(self, data=data, path=path)