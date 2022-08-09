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
from Methods.computeRegSensitivity import computeRegSensitivity
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})


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
        computeSensitivity(dss, initParams)
        computeRegSensitivity(dss, initParams)

    # get init load
    outDSS = getInitDemand(dss, initParams)

    # Dss driver function
    outDSS = dssDriver('InitDSS', dss, outDSS, initParams)

    # Energy scheduling driver function
    outES = schedulingDriver('Dispatch', outDSS, initParams)

    # corrected dss driver function
    outDSS = dssDriver('FinalDSS', dss, outDSS, initParams, outES=outES)

    return outES, outDSS
