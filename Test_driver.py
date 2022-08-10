# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:07:32 2022

@author: Jorge

"""
#########################################################################
from SLP_LP_scheduling import scheduling
import os
import pathlib
import time
import shutil

# get file path
script_path = os.path.dirname(os.path.abspath(__file__))

# init parameters
initParams = dict()
initParams["plot"] = "True"  # plotting flag
initParams["vmin"] = "0.95"  # voltage limits
initParams["vmax"] = "1.05"
initParams["userDemand"] = "None"
initParams["freq"] = "H"  # "15min", "30min", "H"
initParams["dispatchType"] = "SLP"  # "LP"
initParams["script_path"] = str(script_path)  # "LP"
initParams["case"] = "123bus"  # 13bus
initParams["dssFile"] = "IEEE123Master.dss"  # 'IEEE13Nodeckt.dss'

# output directory
# time stamp
t = time.localtime()
timestamp = time.strftime('%b-%d-%Y_%H%M', t)
# create directory to store results
today = time.strftime('%b-%d-%Y', t)
directory = "Results_ncontrol_" + today
output_dir12 = pathlib.Path(script_path).joinpath("outputs", directory)

if not os.path.isdir(output_dir12):
    os.mkdir(output_dir12)

output_dir13 = pathlib.Path(output_dir12).joinpath(initParams["dispatchType"])
if not os.path.isdir(output_dir13):
    os.mkdir(output_dir13)
else:
    shutil.rmtree(output_dir13)
    os.mkdir(output_dir13)
# define traverse parameters
batSizes = [0]
pvSizes = [150]
loadMults = [1]  # 1 for default loadshape, 11 for real.

# main loop
for lm, loadMult in enumerate(loadMults):
    for ba, batSize in enumerate(batSizes):
        for pv, pvSize in enumerate(pvSizes):
            outDir = f"bat_{ba}_pv_{pv}_lm_{lm}"
            print(outDir)
            # define output dir
            output_dir1 = pathlib.Path(output_dir13).joinpath(outDir)
            if not os.path.isdir(output_dir1):
                os.mkdir(output_dir1)
            # store in initParams the moving parameters
            initParams["output_dir"] = str(output_dir1)
            initParams["loadMult"] = str(loadMult)
            initParams["batSize"] = str(batSize)
            initParams["pvSize"] = str(pvSize)

            ####################################
            # 1: compute scheduling
            ####################################
            outES, outDSS = scheduling(initParams)
