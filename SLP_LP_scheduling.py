# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 16:22:51 2021

@author: tefav
"""
# required for processing
import pathlib
import os

import py_dss_interface
import numpy as np
from Methods.dssDriver import dssDriver
from Methods.schedulingDriver import schedulingDriver
from Methods.initDemandProfile import getInitDemand 
from Methods.computeSensitivity import computeSensitivity

#required for plotting
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

def save_initDSS(script_path, Pg_0, v_0, Pjk_0, v_base, Pjk_lim, loadNames, initDemand, initDemandQ):
    # save initial DSS values
    outDSS = dict()
    outDSS['initPower'] = Pg_0
    outDSS['initVolts'] = v_0
    outDSS['initPjks'] = Pjk_0
    outDSS['limPjks'] = Pjk_lim 
    outDSS['nodeBaseVolts'] = v_base 
    outDSS['loadNames'] = loadNames
    outDSS['initDemand'] = initDemand
    outDSS['initDemandQ'] = initDemandQ
    return outDSS

def save_ES(script_path, outGen, outDR, outPsc, outPsd):
    # save optimization values
    outES = dict()
    outES['Gen'] = outGen
    outES['DR'] = outDR
    outES['Pchar'] = outPsc
    outES['Pdis'] = outPsd
    return outES

def SLP_LP_scheduling(loadMult, batSize, pvSize, output_dir, vmin, vmax, userDemand=None, plot=False, freq="15min", dispatchType='SLP'):

    # initialization
    case = '13bus' # 123bus
    file = 'IEEE13Nodeckt.dss'#"IEEE123Master.dss" 

    # execute the DSS model
    script_path = os.path.dirname(os.path.abspath(__file__))
    dss_file = pathlib.Path(script_path).joinpath("EV_data", case, file) 
    
    dss = py_dss_interface.DSSDLL()
    dss.text(f"Compile [{dss_file}]")
    
    #compute sensitivities for the test case
    compute = False
    if compute:
        computeSensitivity(script_path, case, dss, dss_file, plot)
    
    # get init load
    loadNames, dfDemand, dfDemandQ = getInitDemand(script_path, dss, freq, loadMult)
    
    # correct native load by user demand
    if userDemand is not None:
        dfDemand.loc[loadNames.index,:] = userDemand
        
    #Dss driver function
    Pg_0, v_0, Pjk_0, v_base, Pjk_lim = dssDriver(output_dir, 'InitDSS', script_path, case, dss, dss_file, loadNames, dfDemand, dfDemandQ, dispatchType, vmin, vmax, plot=plot)
    outDSS = save_initDSS(script_path, Pg_0, v_0, Pjk_0, v_base, Pjk_lim, loadNames, dfDemand, dfDemandQ)
    
    #Energy scheduling driver function   
    outGen, outDR, outPchar, outPdis, outLMP, costPdr, cgn, mobj = schedulingDriver(batSize, pvSize, output_dir, 'Dispatch', freq, script_path, case, outDSS, dispatchType, vmin, vmax, plot=plot)
    outES = save_ES(script_path, outGen, outDR, outPchar, outPdis)
    
    # normalization 
    loadNames = outDSS['loadNames']
    dfDemand = outDSS['initDemand']
    
    #corrected dss driver function
    Pg, v, Pjk, v_base, Pjk_lim = dssDriver(output_dir, 'FinalDSS', script_path, case, dss, dss_file, loadNames, dfDemand, dfDemandQ, dispatchType, vmin, vmax, out=outES, plot=plot)

    # initial power 
    Pg = np.reshape(Pg.values.T, np.size(Pg), order="F")
    costPg = cgn @ Pg

    operationCost = costPdr[0] + costPg[0]
        
    return dfDemand.loc[loadNames.index,:], outLMP, operationCost, mobj
        
        
