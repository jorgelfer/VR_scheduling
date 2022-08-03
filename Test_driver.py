# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:07:32 2022

@author: Jorge

"""
#########################################################################
from SLP_LP_scheduling import SLP_LP_scheduling
import pandas as pd
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import time
import seaborn as sns
import shutil
from functools import reduce

ext = '.png'
dispatch = 'SLP'
metric = np.inf# 1,2,np.inf
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
    
batSizes = [0]
pvSizes = [0]
loadMults = [1] #1 for default loadshape, 11 for real.

# voltage limits
vmin = 0.95
vmax = 1.06

opcost = np.zeros((len(batSizes), len(pvSizes)))
mopcost = np.zeros((len(batSizes),len(pvSizes)))

for lm, loadMult in enumerate(loadMults): 
    for ba, batSize in enumerate(batSizes): 
        for pv, pvSize in enumerate(pvSizes):
            
            print(f"bat_{ba}_pv_{pv}_lm_{lm}")
            output_dir1 = pathlib.Path(output_dir13).joinpath(f"bat_{ba}_pv_{pv}_lm_{lm}")
            if not os.path.isdir(output_dir1):
                os.mkdir(output_dir1)
                
            ####################################
            # First thing: compute the initial Dispatch
            ####################################
            demandProfile, LMP, OperationCost, mOperationCost = SLP_LP_scheduling(loadMult, batSize, pvSize, output_dir1, vmin, vmax, userDemand=None, plot=plot, freq="30min", dispatchType=dispatch)
            opcost[ba, pv] = OperationCost
            mopcost[ba, pv] = mOperationCost

plt.clf()
fig, axis = plt.subplots(figsize=(h,w))
sns.heatmap(opcost)
plt.tight_layout()
img_path = pathlib.Path(output_dir13).joinpath('heatmap' + ext)
plt.savefig(img_path)
plt.close('all')
            
plt.clf()
fig, axis = plt.subplots(figsize=(h,w))
sns.heatmap(mopcost)
plt.tight_layout()
img_path = pathlib.Path(output_dir13).joinpath('mheatmap' + ext)
plt.savefig(img_path)
plt.close('all')
