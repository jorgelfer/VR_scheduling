# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:07:32 2022

@author: Jorge

"""
#########################################################################
from SLP_LP_scheduling import scheduling
import pandas as pd
import numpy as np
import os
import pathlib
import time
import seaborn as sns
import shutil
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

ext = '.png'
dispatch = 'SLP'
metric = np.inf  # 1,2,np.inf
plot = True
h = 6
w = 4

script_path = os.path.dirname(os.path.abspath(__file__))

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

output_dir13 = pathlib.Path(output_dir12).joinpath(dispatch)
if not os.path.isdir(output_dir13):
    os.mkdir(output_dir13)
else:
    shutil.rmtree(output_dir13)
    os.mkdir(output_dir13)
    
# define traverse parameters
batSizes = [0]
pvSizes = [0]
loadMults = [1]  # 1 for default loadshape, 11 for real.

# voltage limits
vmin = 0.95
vmax = 1.06

# prelocate
opcost = np.zeros((len(batSizes), len(pvSizes)))
mopcost = np.zeros((len(batSizes),len(pvSizes)))

# main loop
for lm, loadMult in enumerate(loadMults):
    for ba, batSize in enumerate(batSizes):
        for pv, pvSize in enumerate(pvSizes):
            outDir = f"bat_{ba}_pv_{pv}_lm_{lm}"
            print(outDir)
            output_dir1 = pathlib.Path(output_dir13).joinpath(outDir)
            if not os.path.isdir(output_dir1):
                os.mkdir(output_dir1)
            ####################################
            # 1: compute scheduling
            ####################################
            demandProfile, LMP, mOperationCost = scheduling(loadMult, batSize, pvSize, output_dir1, vmin, vmax, userDemand=None, plot=plot, freq="30min", dispatchType=dispatch)
