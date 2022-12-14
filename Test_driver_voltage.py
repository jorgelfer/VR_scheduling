# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:07:32 2022

@author: Jorge

"""
#########################################################################
import shutil
import time
import os
import pathlib
import numpy as np
from Methods.sensitivityPy import sensitivityPy
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import py_dss_interface


# set control mode off
def set_baseline(dss):
    dss.text("Set Maxiterations=100")
    dss.text("Set controlmode=OFF")

ext = '.pdf'
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
directory = "Results_linearity_" + today
output_dir = pathlib.Path(script_path).joinpath("outputs", directory)

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
# else:
#     shutil.rmtree(output_dir)
#     os.mkdir(output_dir)

# voltage limits
vmin = 0.95
vmax = 1.05

# initialization
dssCase = '123bus'  # 123bus
file = "IEEE123Master.dss"  # 'IEEE13Nodeckt.dss', "IEEE123Master.dss"

# get file path
script_path = os.path.dirname(os.path.abspath(__file__))
dssFile = pathlib.Path(script_path).joinpath("EV_data", dssCase, file)

# execute the DSS model
dss = py_dss_interface.DSSDLL()
dss.text(f"Compile [{dssFile}]")

# get node names
nodeNames = dss.circuit_all_node_names()

# create sensitivityPy object
sen = sensitivityPy(dss, time=0)

# get all node-based base volts
nodeBaseVolts_series = sen.get_nodeBaseVolts()
nodeBaseVolts = [nodeBaseVolts_series[n] for n in nodeNames]

# get all node-based lines names
nodeLineNames, lines = sen.get_nodeLineNames()

# list dss elements
trafos = dss.transformers_all_Names()
regs = [tr for tr in trafos if "reg" in tr]

outputFile = pathlib.Path(output_dir).joinpath("TapVoltage.npy")
if not os.path.isfile(outputFile):
    # prelocate
    numTaps = 32
    numReg = len(regs) 
    volts = np.zeros((len(nodeNames), numTaps + 1, numReg))
    flows = np.zeros((len(nodeLineNames), numTaps + 1, numReg))
    # main loop
    for r, reg in enumerate(regs):
        print(f"{reg}")
        ntap = 0
        for tap in range(-16, 17):
            # fresh compilation: to remove previous modifications
            dss.text(f"Compile [{dssFile}]")
            set_baseline(dss)
            # create a sensitivity object
            sen_obj = sensitivityPy(dss, time=0)
            # Perturb DSS with tap change
            sen_obj.perturbRegDSS(reg, 1.0 + 0.00625 * tap)  # +1 tap
            # solve dss file
            dss.text("solve")
            # extract voltage
            volts[:, ntap, r] = sen_obj.voltageMags()
            # extract flows
            flows[:, ntap, r], _, _, _ = sen_obj.flows(nodeLineNames)
            ntap += 1

    # save output file
    with open(outputFile, 'wb') as f:
        np.save(f, volts)
        np.save(f, flows)
else:
    with open(outputFile, 'rb') as f:
        volts = np.load(f)
        flows = np.load(f)

# flows 
for c, cond in enumerate(nodeLineNames):
    print(f"{cond}")
    # create a new directory for each nod
    output_dirCond = pathlib.Path(output_dir).joinpath(f"{cond}")
    if not os.path.isdir(output_dirCond):
        os.mkdir(output_dirCond)

    for r, reg in enumerate(regs):
        flowMin = flows[c, :, :].min()
        flowMax = flows[c, :, :].max()
        # compute pu
        flow = flows[c, :, r]
        # create plot
        plt.clf()
        fig, ax = plt.subplots()  # figsize=(h,w)
        tapRange = np.arange(-16, 17, 1)
        plt.scatter(tapRange, flow, s=6)
        plt.ylim(0.99 * flowMin, 1.01 * flowMax)
        plt.title(f"{cond}_{reg}")
        fig.tight_layout()
        output_img = pathlib.Path(output_dirCond).joinpath(f"flow_{cond}_{reg}" + ext)
        plt.savefig(output_img)
        plt.close('all')

# voltages
for n, node in enumerate(nodeNames):
    print(f"{node}")
    # create a new directory for each nod
    output_dirNode = pathlib.Path(output_dir).joinpath(f"{node}")
    if not os.path.isdir(output_dirNode):
        os.mkdir(output_dirNode)
    # main loop
    for r, reg in enumerate(regs):
        # compute pu
        vpu = volts[n, :, r] / (1000 * nodeBaseVolts[n])
        vpu = np.expand_dims(vpu, axis=1)
        vmin_vec = vmin * np.zeros((len(vpu), 1))
        vmax_vec = vmax * np.zeros((len(vpu), 1))
        concat = np.hstack((vmin_vec, vmax_vec))
        # create plot
        plt.clf()
        fig, ax = plt.subplots()  # figsize=(h,w)
        tapRange = np.arange(-16, 17, 1)
        plt.scatter(tapRange, vpu, s=6)
        plt.plot(tapRange, concat)
        plt.ylim(0.99 * vmin, 1.01 * vmax)
        plt.title(f"{node}_{reg}")
        fig.tight_layout()
        output_img = pathlib.Path(output_dirNode).joinpath(f"voltage_{node}_{reg}" + ext)
        plt.savefig(output_img)
        plt.close('all')
