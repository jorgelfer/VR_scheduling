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
    

def plot_sensi(dfVolts, dfFlows, cond, path, dV_min, dV_max, dPjk_min, dPjk_max):
    ext = '.pdf'
    # VoltageSensitivity
    plt.clf()
    fig, ax = plt.subplots()
    ax = sns.heatmap(dfVolts, annot=False, vmin=dV_min, vmax=dV_max)
    fig.tight_layout()
    output_img = pathlib.Path(path).joinpath("VoltageSensitivity_" + cond + ext)
    plt.savefig(output_img)
    plt.close('all')
    # PTDF
    plt.clf()
    fig, ax = plt.subplots()
    ax = sns.heatmap(dfFlows, annot=False, vmin=dPjk_min, vmax=dPjk_max)
    fig.tight_layout()
    output_img = pathlib.Path(path).joinpath("PTDF_" + cond + ext)
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

# load sensitivity of quantities to reg
dVdR = pd.read_pickle(pathlib.Path(script_path).joinpath("inputs", dssCase, "VoltageToRegSensitivity.pkl"))
dPjkdR = pd.read_pickle(pathlib.Path(script_path).joinpath("inputs", dssCase, "FlowsToRegSensitivity.pkl"))

# load or compute reg affected sensitivities
dVdP_path = pathlib.Path(script_path).joinpath("inputs", dssCase, "VoltageSensitivity.pkl")
dPjkdP_path = pathlib.Path(script_path).joinpath("inputs", dssCase, "PTDF.pkl")

# losses
ckt_losses = dss.circuit_losses()
if not os.path.isfile(dVdP_path) or not os.path.isfile(dPjkdP_path):
    # prelocate
    dPjkdP_a = np.zeros([len(nodeLineNames), len(nodeNames)])
    dVdP_a = np.zeros([len(nodeNames), len(nodeNames)])
    # main loop through all nodes
    for n, node in enumerate(nodeNames):
        # fresh compilation to remove previous modifications
        dss.text(f"Compile [{dssFile}]")
        set_baseline(dss)

        # create a sensitivity object
        sen_obj = sensitivityPy(dss, time=0)

        # Perturb DSS with small gen
        sen_obj.perturbDSS(node, kv=nodeBaseVolts[node], kw=10, P=True)  # 10 kw

        dss.text("solve")

        # compute Voltage sensitivity
        currVolts, _ = sen_obj.voltageProfile()
        dVdP_a[:, n] =  currVolts- initVolts

        # compute PTDF
        currPjk, _, _, _ = sen_obj.flows(nodeLineNames)
        dPjkdP_a[:, n] = currPjk - initPjk

    # save
    dVdP = pd.DataFrame(dVdP_a, np.asarray(nodeNames), np.asarray(nodeNames))
    dVdP.to_pickle(dVdP_path)
    dPjkdP = pd.DataFrame(dPjkdP_a, np.asarray(nodeLineNames), np.asarray(nodeNames))
    dPjkdP.to_pickle(dPjkdP_path)

else:
    dVdP = pd.read_pickle(dVdP_path)
    dPjkdP = pd.read_pickle(dPjkdP_path)

##############################################################################
# 2. compute regulator affected sensitivities
##############################################################################

for reg in regs:
    print(f"{reg}")

    # load or compute reg affected sensitivities
    dVdRdP_path = pathlib.Path(script_path).joinpath("inputs", dssCase, f"VoltageSensitivity_{reg}.pkl")
    dPjkdRdP_path = pathlib.Path(script_path).joinpath("inputs", dssCase, f"PTDF_{reg}.pkl")

    if not os.path.isfile(dVdRdP_path) or not os.path.isfile(dPjkdRdP_path):
        # prelocate
        dVdRdP_a = np.zeros([len(nodeNames), len(nodeNames)])
        dPjkdRdP_a = np.zeros([len(nodeLineNames), len(nodeNames)])
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
            dVdRdP_a[:, n] =  currVolts- initVolts

            # compute PTDF
            currPjk, _, _, _ = sen_obj.flows(nodeLineNames)
            dPjkdRdP_a[:, n] = currPjk - initPjk

        # save
        dVdRdP = pd.DataFrame(dVdRdP_a, np.asarray(nodeNames), np.asarray(nodeNames))
        dVdRdP.to_pickle(dVdRdP_path)
        dPjkdRdP = pd.DataFrame(dPjkdRdP_a, np.asarray(nodeLineNames), np.asarray(nodeNames))
        dPjkdRdP.to_pickle(dPjkdRdP_path)

    else:
        dVdRdP = pd.read_pickle(dVdRdP_path)
        dPjkdRdP = pd.read_pickle(dPjkdRdP_path)

    # directory to store plots
    output_dirReg = pathlib.Path(output_dir).joinpath(f"{reg}")
    if not os.path.isdir(output_dirReg):
        os.mkdir(output_dirReg)

    # new dataframes
    diff_dV = dVdRdP - dVdP
    diff_dPjk = dPjkdRdP - dPjkdP

    newdV = dVdP.add(dVdR.loc[:, reg], axis=0)
    newdPjk = dPjkdP.add(dPjkdR.loc[:, reg], axis=0)

    new_diff_dV = dVdRdP - newdV
    new_diff_dPjk = dPjkdRdP - newdPjk

    # np.linalg.norm(diff_dV, ord=np.inf)
    # np.linalg.norm(diff_dPjk, ord=np.inf) # 1,2,np.inf
    # np.linalg.norm(new_diff_dV, ord=np.inf)
    # np.linalg.norm(new_diff_dPjk, ord=np.inf) # 1,2,np.inf

    # mins for plotting
    dV_min = min([dVdP.min().min(), dVdRdP.min().min(),
                  diff_dV.min().min(), new_diff_dV.min().min()])
    dV_max = max([dVdP.max().max(), dVdRdP.max().max(),
                  diff_dV.max().max(), new_diff_dV.max().max()])
    dPjk_min = min([dPjkdP.min().min(), dPjkdRdP.min().min(),
                  diff_dPjk.min().min(), new_diff_dPjk.min().min()])
    dPjk_max = max([dPjkdP.max().max(), dPjkdRdP.max().max(),
                  diff_dPjk.max().max(), new_diff_dPjk.max().max()])

    # plot
    plot_sensi(dVdP, dPjkdP, "dxdP", output_dirReg, dV_min, dV_max, dPjk_min, dPjk_max)

    plot_sensi(dVdRdP, dPjkdRdP, "dxdRdP", output_dirReg, dV_min, dV_max, dPjk_min, dPjk_max)

    plot_sensi(diff_dV, diff_dPjk, "diff", output_dirReg, dV_min, dV_max, dPjk_min, dPjk_max)

    plot_sensi(new_diff_dV, new_diff_dPjk, "newDiff", output_dirReg, dV_min, dV_max, dPjk_min, dPjk_max)
