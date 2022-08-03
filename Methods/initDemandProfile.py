# -*- coding: utf-8 -*-
"""
Created on 06/03/2022
@author:Jorge 


Extract from opendss the native load and multiply it by a load shape
"""
import pandas as pd
import pathlib
from Methods.loadHelper import loadHelper
import numpy as np

def get_1ph_demand(dss, mv_node_name):
    "Method to extract the demand from a feeder"
    # create a dictionary from node names
    load_power_dict = {key: 0 for key in mv_node_name}
    load_Q_dict = {key: 0 for key in mv_node_name}
    loadNameDict = dict()
    
    elems = dss.circuit_all_element_names()
    for i, elem in enumerate(elems):
        dss.circuit_set_active_element(elem)
        if "Load" in elem:
            
            # extract load name
            loadName = elem.split(".")[1]
            
            # write load name
            dss.loads_write_name(loadName)
            
            # get bus name  
            buses = dss.cktelement_read_bus_names()
            bus = buses[0]
            
            # save name
            loadNameDict[bus] = loadName
            
            # save kw
            load_power_dict[bus] = dss.loads_read_kw()
            load_Q_dict[bus] = dss.loads_read_kvar()
            
    return pd.Series(loadNameDict), pd.Series(load_power_dict), pd.Series(load_Q_dict)         

def load_hourlyDemand(scriptPath, numLoads, freq):
    hourlyDemand_file = pathlib.Path(scriptPath).joinpath("inputs", "HourlyDemands100.xlsx")
    t = pd.read_excel(hourlyDemand_file, sheet_name='august14') # extract demand for august 14 - 2018.
    
    t = t.set_index('Hour')

    # create load helper method
    help_obj = loadHelper(initfreq = 'H', finalFreq = freq)

    # call method for processing series
    dfDemand = help_obj.process_pdFrame(t)
    
    return dfDemand

def load_GenerationMix(script_path, freq):

    GenMix_file = pathlib.Path(script_path).joinpath("inputs", "GeorgiaGenerationMix2.xlsx")
    t = pd.read_excel(GenMix_file)
    
    # create load helper method
    help_obj = loadHelper(initfreq = 'H', finalFreq = freq)
    
    # load genalpha with interpolation
    genAlpha = t["Alpha"]
    # call method for processing series
    genAlpha = help_obj.process_pdSeries(genAlpha)
    
    # load genbeta with interpolation
    genBeta = t["Beta"] # load shape for 2018-08-14
    # call method for processing series
    genBeta = help_obj.process_pdSeries(genBeta)
    
    return genAlpha, genBeta


def getInitDemand(scriptPath, dss, freq, loadMult):

    # get all node-based buses, 
    nodeNames = dss.circuit_all_node_names()
    
    # get native load
    loadNames, loadKws, loadKvars = get_1ph_demand(dss, nodeNames)
    
    # get native loadshape, 
    _, genBeta = load_GenerationMix(scriptPath, freq)

    # expand dims of native load
    demandProfile = loadMult * loadKws.to_frame()
    demandQrofile = loadMult * loadKvars.to_frame()
    
    # Expand feeder demand for time series analysis
    demandProfile =  demandProfile.values @ genBeta.values.T # 2018-08-14
    demandQrofile =  demandQrofile.values @ genBeta.values.T # 2018-08-14

    # Active Power df 
    dfDemand = pd.DataFrame(demandProfile) 
    dfDemand.index = loadKws.index 
    dfDemand.columns = genBeta.index.strftime('%H:%M')

    # get real load
    #############
    
    #realDemand = load_hourlyDemand(scriptPath, len(loadNames), freq)
    #realDemand = realDemand.T
    #realDemand = realDemand[:len(loadNames)]
    #realDemand = loadMult*realDemand
    #dfDemand.loc[loadNames.index,:] = realDemand.values

    # Reactive Power df
    # if fixed power factor:
    # random power factors
    #np.random.seed(2022)
    #PF = np.random.uniform(0.85, 1, size=len(dfDemand.index))
    #dfDemandQ = (np.tan(np.arccos(PF)) * dfDemand.T).T
    # else:
    dfDemandQ = pd.DataFrame(demandQrofile) 
    dfDemandQ.index = loadKvars.index 
    dfDemandQ.columns = genBeta.index.strftime('%H:%M')

    return loadNames, dfDemand, dfDemandQ
