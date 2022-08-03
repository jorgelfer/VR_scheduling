# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 09:43:57 2021

@author: tefav
"""
import numpy as np
import pandas as pd
import math

class sensitivityPy:

    def __init__(self, dss, time):
        self.dss = dss
        self.time = time
        
    # Methods
    def perturbDSS(self, node, kv, kw, P):
        genName = node.replace(".","_")
        if P:
            self.__new_1ph_gen(genName, node, kv, kw, kvar=0)
        else:
            self.__edit_1ph_gen(genName, node, kw=0, kvar=kw)

    def setLoads(self, loadP, loadQ, loadNames):
        
        # rename nodes to explicit load names
        loadP = loadP.loc[loadNames.index].rename(index=loadNames)
        loadQ = loadQ.loc[loadNames.index].rename(index=loadNames)
                
        # modify loads
        self.__setAllLoads(loadP, loadQ)
        
    def modifyDSS(self, out, baseVolts):
        
        # extract dispatch results
        Pg, Pdr, Pchar, Pdis = self.__extractOutput(out)
        
        # modify loads
        if Pdr.any(axis=None):
            self.__modifyAllLoads(Pdr)
        
        # create PVs
        if Pg is not None:
            self.__createPG(Pg, baseVolts)

        # create Storage discharge
        if Pdis is not None:
            self.__createPG(Pdis, baseVolts)

        # create Storage charge 
        if Pchar is not None:
            self.__createLoad(Pchar)

    def voltageProfile(self):
        
        # debug
        # get all node magnitudes
        # voltages = self.dss.circuit_all_bus_vmag()
        # nodeNames = self.dss.circuit_all_node_names()
        
        # values
        volts = self.dss.circuit_all_bus_volts()
                
        vReal = volts[0::2]
        vImaginary = volts[1::2]
                
        vphasor = np.asarray(vReal) + 1j*np.asarray(vImaginary)
        vphasor_mag = abs(vphasor)
        
        return np.asarray(vphasor_mag), np.asarray(vphasor)
    
    def flows(self, lname):
          
        # get line flows
        lines, lines_KW_power, lines_KVAR_power = self.__getLinePowers()
        
        # organize line flows for node-based analysis
        Pjk, Qjk, pjk, qjk = self.__organizeFlows(lines, lines_KW_power, lines_KVAR_power, lname)

        return Pjk, Qjk, pjk, qjk

    def get_nodeBaseVolts(self):
        
        # create a directory to store voltages
        node_base_voltage = dict()
        
        # get all buses
        buses = self.dss.circuit_all_bus_names()
        
        # iterate over buses
        for i, bus in enumerate(buses):
            self.dss.circuit_set_active_bus(bus)
            bus_voltage = self.dss.bus_kv_base() # VLN for 1phase buses
            bus_nodes = self.dss.bus_nodes()
            for n in bus_nodes:
                node = bus + f'.{n}'
                node_base_voltage[node] = bus_voltage
        
        return pd.Series(node_base_voltage)

    def get_nodeLineNames(self):
        # phase-based names
        lname = list()
        lines = list()
        elements = self.dss.circuit_all_element_names()
        
        for i, elem in enumerate(elements):
            self.dss.circuit_set_active_element(elem)
            if "Line" in elem:
                # line name
                lines.append(elem)
                # get node-based line names
                buses = self.dss.cktelement_read_bus_names()
                # get this element node and discard the reference
                nodes = [i for i in self.dss.cktelement_node_order() if i != 0]
                # reorder the number of nodes
                nodes = np.asarray(nodes).reshape((int(len(nodes)/2),-1),order="F")                
                
                for t1n, t2n in zip(nodes[:,0],nodes[:,1]):
                    lname.append("L"+ buses[0].split(".")[0] + f".{t1n}" + "-" + buses[1].split(".")[0] + f".{t1n}")
    
            elif "Transformer" in elem:
                # line name
                lines.append(elem)
                # get node-based transformer names
                buses = self.dss.cktelement_read_bus_names()
                # get this element node and discard the reference
                nodes = [i for i in self.dss.cktelement_node_order() if i != 0]
                # reorder the number of nodes
                nodes = np.asarray(nodes).reshape((int(len(nodes)/2),-1),order="F")
            
                for t1n, t2n in zip(nodes[:,0],nodes[:,1]):
                    lname.append("T"+ buses[0].split(".")[0] + f".{t1n}" + "-" + buses[1].split(".")[0] + f".{t1n}")
    
        return lname, lines
    
    # Helper methods
    
    def __extractOutput(self, out):
        """Method to define outputs"""
    
        Pg   = out['Gen'] 
        Pdr  = out['DR']   
        Pchar= out['Pchar']  
        Pdis = out['Pdis']  
    
        return Pg, Pdr, Pchar, Pdis 
    
    def __organizeFlows(self, lines, lines_KW_power, lines_KVAR_power, lname):

        Pjk = np.zeros([len(lname)]) # containing flows Pkm
        Pkj = np.zeros([len(lname)]) # containing flows Pmk

        Qjk = np.zeros([len(lname)]) # containing flows Pkm
        Qkj = np.zeros([len(lname)]) # containing flows Pmk
        
        # aggregated per line
        pjk = np.zeros([len(lines)]) # containing flows Pkm
        qjk = np.zeros([len(lines)]) # containing flows Pkm

        cont = 0
        for l, line in enumerate(lines):

            p = np.asarray(lines_KW_power[line])
            q = np.asarray(lines_KVAR_power[line])

            Pjk[cont: cont + int(len(p)/2)] = p[:int(len(p)/2)]
            Pkj[cont: cont + int(len(p)/2)] = p[int(len(p)/2):]

            Qjk[cont: cont + int(len(p)/2)] = q[:int(len(p)/2)]
            Qkj[cont: cont + int(len(p)/2)] = q[int(len(p)/2):]

            # total line flow 
            pjk[l] = sum(p[:int(len(p)/2)])
            qjk[l] = sum(q[:int(len(p)/2)])

            cont += int(len(p)/2)

        return Pjk, Qjk, pjk, qjk

    def __getLinePowers(self):

        # prelocate 
        lines = list()
        lines_KW_dict = dict()
        lines_KVAR_dict = dict()
    
        elements = self.dss.circuit_all_element_names()
    
        for i, elem in enumerate(elements):
            self.dss.circuit_set_active_element(elem)
            if "Line" in elem:
                lines.append(elem)
                lines_KW_dict[elem] = self.dss.cktelement_powers()[0::2]
                lines_KVAR_dict[elem] = self.dss.cktelement_powers()[1::2]

            elif "Transformer" in elem:
                lines.append(elem)
                
                if elem == 'Transformer.xfm1':
                    lines_KW_dict[elem] = self.dss.cktelement_powers()[0::2]
                    lines_KVAR_dict[elem] = self.dss.cktelement_powers()[1::2]
                    del lines_KW_dict[elem][3]
                    del lines_KW_dict[elem][-1]
                    del lines_KVAR_dict[elem][3]
                    del lines_KVAR_dict[elem][-1]
                else: 
                    lines_KW_dict[elem] = [i for i in self.dss.cktelement_powers()[0::2] if i != 0]
                    lines_KVAR_dict[elem] = [i for i in self.dss.cktelement_powers()[1::2] if i != 0]

        return lines, lines_KW_dict, lines_KVAR_dict

    def __modifyAllLoads(self, Pdr):
        "Method to modify loads from a DSS file according to dispatch"
        
        elems = self.dss.circuit_all_element_names()
        for i, elem in enumerate(elems):
            self.dss.circuit_set_active_element(elem)
            if "Load" in elem:
                
                # extract load name
                loadName = elem.split(".")[1]
                
                # check if dispatch allocated DR for this load
                if Pdr.loc[loadName,self.time] != 0:
                    # write load name
                    self.dss.loads_write_name(loadName)
                    # compute new demand
                    newKw = self.dss.loads_read_kw() - Pdr.loc[loadName,self.time]
                    # read kvar
                    kvar = self.dss.loads_read_kvar()                
                    # save kw
                    self.__modifyLoad(newKw, kvar,  loadName)
                    

                    
    def __setAllLoads(self, instDemandP, instDemandQ):
        "Method to modify loads from a DSS file according to dispatch"
        
        elems = self.dss.circuit_all_element_names()
        for i, elem in enumerate(elems):
            self.dss.circuit_set_active_element(elem)
            if "Load" in elem:   
                # extract load name
                loadName = elem.split(".")[1]
                # write load name
                self.dss.loads_write_name(loadName)
                # read kvar
                kvar = instDemandQ[loadName]
                # save kw
                kw = instDemandP[loadName]
                if math.isnan(kw):
                    breakpoint()
                self.__modifyLoad(kw, kvar,  loadName)

    def __modifyLoad(self, kw, kvar, load):
        self.dss.text(f"edit load.{load} "
                      f"kw={kw} "
                      f"kvar={kvar}")
        
    def __createPG(self, Pg, gen_kvs):
        "Method to modify loads from a DSS file according to dispatch"
        
        for node in Pg.index:
            # check if dispatch allocated power for this PV at this hour
            if Pg.loc[node, self.time] != 0:
                genName = node.replace(".","_")
                self.__new_1ph_gen(genName, node, gen_kvs[node], Pg.loc[node, self.time], kvar=0)
        
    def __new_1ph_gen(self, gen, node, kv, kw, kvar):
        self.dss.text(f"new generator.{gen} "
                      f"phases=1 "
                      f"kv={kv} "
                      f"bus1={node} "
                      f"kw={kw} "
                      f"kvar={kvar}")

    def __edit_1ph_gen(self, gen, kw, kvar):
        self.dss.text(f"edit generator.{gen} "
                      f"kw={kw} "
                      f"kvar={kvar}")

    def __createLoad(self, Pchar):
        "Method to modify loads from a DSS file according to dispatch"
        
        for node in Pchar.index:
            # check if dispatch allocated power for this PV at this hour
            if Pchar.loc[node, self.time] != 0:
                loadName = node.replace(".","_")
                self.__new_1ph_load(loadName, node, Pchar.loc[node, self.time])
    
    def __new_1ph_load(self, loadName, node, kw):
        self.dss.text(f"new load.{loadName} "
                      f"bus1={node} "
                      f"phases=1 "
                      f"conn=wye "
                      f"model=1 "
                      f"kw={kw} "
                      f"pf=1")

