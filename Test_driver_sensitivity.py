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
import seaborn as sns
import pandas as pd
from Methods.sensitivityPy import sensitivityPy
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import py_dss_interface


# set control mode off
def set_baseline(dss):
    dss.text("Set Maxiterations=100")
    dss.text("Set controlmode=OFF")
    


def plot_sensi(dfVolts, dfFlows, cond, reg, script_path):
    h = 20
    w = 20
    ext = '.png'
    # VoltageSensitivity
    plt.clf()
    fig, ax = plt.subplots(figsize=(h, w))
    ax = sns.heatmap(dfVolts, annot=False)
    fig.tight_layout()
    output_img = pathlib.Path(script_path).joinpath("outputs", f"VoltageSensitivity_{reg}" + cond + ext)
    plt.savefig(output_img)
    plt.close('all')
    # PTDF
    plt.clf()
    fig, ax = plt.subplots(figsize=(h, w))               
    ax = sns.heatmap(dfFlows, annot=False)
    fig.tight_layout()
    output_img = pathlib.Path(script_path).joinpath("outputs", f"PTDF_{reg}" + cond + ext)
    plt.savefig(output_img)
    plt.close('all')


script_path = os.path.dirname(os.path.abspath(__file__))
# output directory
# time stamp
t = time.localtime()
timestamp = time.strftime('%b-%d-%Y_%H%M', t)
# create directory to store results
today = time.strftime('%b-%d-%Y', t)
directory = "Results_Flows" + today
output_dir = pathlib.Path(script_path).joinpath("outputs", directory)

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

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
set_baseline(dss)
dss.text("solve")

# get node names
nodeNames = dss.circuit_all_node_names()

# create sensitivityPy object
sen = sensitivityPy(dss, time=0)

# get all node-based base volts
nodeBaseVolts = sen.get_nodeBaseVolts()

# get all node-based lines names
nodeLineNames, lines = sen.get_nodeLineNames()

# list dss elements
trafos = dss.transformers_all_Names()
regs = [tr for tr in trafos if "reg" in tr]

# init Volts
initVolts = sen.voltageMags()

# get base pjk
initPjk, _, _, _ = sen.flows(nodeLineNames)

# list DSS regulators
trafos = dss.transformers_all_Names()
regs = [tr for tr in trafos if "reg" in tr]

# base sensitivity
dfV_init = pd.read_pickle(pathlib.Path(script_path).joinpath("inputs", dssCase, "VoltageSensitivity.pkl"))
dV_dr = pd.read_pickle(pathlib.Path(script_path).joinpath("inputs", dssCase, "VoltageToRegSensitivity.pkl"))
dfPjk_init = pd.read_pickle(pathlib.Path(script_path).joinpath("inputs", dssCase, "PTDF.pkl"))
dPjk_dr = pd.read_pickle(pathlib.Path(script_path).joinpath("inputs", dssCase, "FlowsToRegSensitivity.pkl"))

for reg in regs:
    print(f"{reg}")
    dPjk = np.zeros([len(nodeLineNames), len(nodeNames)])
    dV = np.zeros([len(nodeNames), len(nodeNames)])
    # main loop through all nodes
    for n, node in enumerate(nodeNames):
        # fresh compilation to remove previous modifications
        dss.text(f"Compile [{dssFile}]")
        set_baseline(dss)

        # create a sensitivity object
        sen_obj = sensitivityPy(dss, time=0)

        # perturb by changing a tap
        sen_obj.perturbRegDSS(reg, 1.0 + 0.00625)  # +1 tap

        # Perturb DSS with small gen
        sen_obj.perturbDSS(node, kv=nodeBaseVolts[node], kw=10, P=True)  # 10 kw

        dss.text("solve")

        # compute Voltage sensitivity
        currVolts, _ = sen_obj.voltageProfile()
        dV[:, n] =  currVolts- initVolts

        # compute PTDF
        currPjk, _, _, _ = sen_obj.flows(nodeLineNames)
        dPjk[:, n] = currPjk - initPjk

    # save
    dfVS = pd.DataFrame(dV, np.asarray(nodeNames), np.asarray(nodeNames))
    dfVS.to_pickle(pathlib.Path(script_path).joinpath("inputs", dssCase, "VoltageSensitivity_{reg}.pkl"))
    dfPjk = pd.DataFrame(dPjk, np.asarray(nodeLineNames), np.asarray(nodeNames))
    dfPjk.to_pickle(pathlib.Path(script_path).joinpath("inputs", dssCase, "PTDF_{reg}.pkl"))

    plot_sensi(dfVS, dfPjk, "ori", reg, script_path)

    # plot
    plot_sensi(dfVS - dfV_init, dfPjk - dfPjk_init, "diff", reg, script_path)

    # compute difference
    newdV = dfV_init.add(dV_dr.loc[:, reg], axis=0)
    newdPjk = dfPjk_init.add(dPjk_dr.loc[:, reg], axis=0)
    plot_sensi(dfVS - newdV, dfPjk - newdPjk, "add-diff", reg, script_path)
