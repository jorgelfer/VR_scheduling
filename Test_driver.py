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
import numpy as np
import pandas as pd

# get file path
script_path = os.path.dirname(os.path.abspath(__file__))

# init parameters
initParams = dict()
initParams["plot"] = "True"  # plotting flag
initParams["vmin"] = "0.96"  # voltage limits
initParams["vmax"] = "1.04"
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
directory = "Results_newSensi_maxOutput2_" + today
output_dir12 = pathlib.Path(script_path).joinpath("outputs", directory)

if not os.path.isdir(output_dir12):
    os.mkdir(output_dir12)
else:
    shutil.rmtree(output_dir12)
    os.mkdir(output_dir12)

# define traverse parameters
regFlags = ["False", "True"]
loadMults = [13]  # 1 for default loadshape, 11 for real.
batSizes = [0]
pvSizes = [200, 300, 400]

DRcost = np.zeros((len(regFlags), len(loadMults), len(batSizes), len(pvSizes)))
J = np.zeros((len(regFlags), len(loadMults), len(batSizes), len(pvSizes)))
losses_cost = np.zeros((len(regFlags), len(loadMults), len(batSizes), len(pvSizes)))
maxOutput = list()
# main loop
for fl, flag in enumerate(regFlags):
    for lm, loadMult in enumerate(loadMults):
        for ba, batSize in enumerate(batSizes):
            for pv, pvSize in enumerate(pvSizes):
                outDir = f"reg_{flag}_bat_{ba}_pv_{pv}_lm_{lm}"
                print(outDir)
                # define output dir
                output_dir1 = pathlib.Path(output_dir12).joinpath(outDir)
                if not os.path.isdir(output_dir1):
                    os.mkdir(output_dir1)
                # store in initParams the moving parameters
                initParams["reg"] = flag
                initParams["output_dir"] = str(output_dir1)
                initParams["loadMult"] = str(loadMult)
                initParams["batSize"] = str(batSize)
                initParams["pvSize"] = str(pvSize)
    
                ####################################
                # compute scheduling
                ####################################
                outES, outDSS = scheduling(initParams)
                DRcost[fl, lm, ba, pv] = outES["DRcost"]
                J[fl, lm, ba, pv] = outES["J"]
                losses_cost[fl, lm, ba, pv] = outDSS["losses_cost"]
                maxOutput.append(outES["maxOutput"])

npyFile = pathlib.Path(output_dir12).joinpath('cost.npy')
with open(npyFile, 'wb') as f:
    np.save(f, J)
    
npyFile2 = pathlib.Path(output_dir12).joinpath('DRcost.npy')
with open(npyFile2, 'wb') as f:
    np.save(f, DRcost)

npyFile3 = pathlib.Path(output_dir12).joinpath('losses_cost.npy')
with open(npyFile3, 'wb') as f:
    np.save(f, losses_cost)

pklFile = pathlib.Path(output_dir12).joinpath('maxOutput.pkl')
pd_concat = pd.concat(maxOutput, axis=1)
pd_concat.to_pickle(str(pklFile))
