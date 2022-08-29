# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 16:22:51 2021

@author: tefav
"""
# required for processing
import pathlib
import py_dss_interface

from Methods.dssDriver import dssDriver
from Methods.schedulingDriver import schedulingDriver
from Methods.initDemandProfile import getInitDemand 
from Methods.computeSensitivity import computeSensitivity
from Methods.computeRegSensitivity_flows import computeRegSensitivity
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})


def compute_violatingVolts(outDSS, initParams):
    # preprocess
    v_0 = outDSS['initVolts']
    v_base = outDSS['nodeBaseVolts']
    vmin = float(initParams["vmin"])
    vmax = float(initParams["vmax"])
    dfDemand = outDSS["dfDemand"]
    # define load nodes
    indexDemand = dfDemand.any(axis=1)
    v = v_0[indexDemand.values].divide((1000 * v_base[indexDemand.values]), axis=0)
    # extract violating Lines
    compare = (v > vmax) | (v < vmin)
    violatingVolts = compare.any(axis=1)
    return violatingVolts


def compute_violatingLines(outDSS):
    '''function to load line Limits'''
    Pij = outDSS['initPjks']
    Pjk_lim = outDSS['limPjks']

    compare = Pij > Pjk_lim
    violatingLines = compare.any(axis=1)
    return violatingLines


def scheduling(initParams):
    # preprocess
    script_path = initParams["script_path"]
    case = initParams["case"]
    file = initParams["dssFile"]

    # execute the DSS model
    dss_file = pathlib.Path(script_path).joinpath("EV_data", case, file)
    dss = py_dss_interface.DSSDLL()
    dss.text(f"Compile [{dss_file}]")

    # compute sensitivities for the test case
    compute = False
    if compute:
        # computeSensitivity(dss, initParams)
        computeRegSensitivity(dss, initParams)
    # get init load
    outDSS = getInitDemand(dss, initParams)

    # Dss driver function
    outDSS = dssDriver('InitDSS', dss, outDSS, initParams)

    # compute violating lines
    violatingVolts = compute_violatingVolts(outDSS, initParams)
    violatingLines = compute_violatingLines(outDSS)

    # initialize
    outES = None
    k = 0
    maxIter = 1 

    while (violatingVolts.any() or violatingLines.any()) and k < maxIter:

        outDSS['violatingVolts'] = violatingVolts.any()
        outDSS['violatingLines'] = violatingVolts.any()

        # Energy scheduling driver function
        outES = schedulingDriver('Dispatch', outDSS, initParams, outES)

        # corrected dss driver function
        outDSS = dssDriver('FinalDSS', dss, outDSS, initParams, outES=outES)

        violatingVolts = compute_violatingVolts(outDSS, initParams)
        violatingLines = compute_violatingLines(outDSS)
        
        k += 1

    return outES, outDSS, k
