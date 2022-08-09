"""
# -*- coding: utf-8 -*-
# @Time    : 10/11/2021 6:09 PM
# @Author  : Jorge Fernandez
"""

from Methods.sensitivityPy import sensitivityPy 
import numpy as np
import pandas as pd
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt


def set_baseline(dss):
    dss.text("Set Maxiterations=100")
    dss.text("Set controlmode=Off")  # disabling regulators


def computeRegSensitivity(dss, initParams):
    # preprocess
    dss_file = initParams["dssFile"]
    case = initParams["case"]
    script_path = initParams["script_path"]
    # initial DSS execution
    dss.text(f"Compile [{dss_file}]")
    set_baseline(dss)
    dss.text("solve")
    # create a sensitivity object
    sen_obj = sensitivityPy(dss, time=0)
    # get all node-based base volts
    nodeBaseVoltage = sen_obj.get_nodeBaseVolts()
    # get all node-based buses
    nodeNames = dss.circuit_all_node_names()
    # get all node-based lines names
    nodeLineNames, lines = sen_obj.get_nodeLineNames()
    # get base voltage
    baseVolts = sen_obj.voltageMags()
    # get base pjk
    basePjk, _, _, _ = sen_obj.flows(nodeLineNames)
    # list DSS regulators
    trafos = dss.transformers_all_Names()
    regs = [tr for tr in trafos if "reg" in tr]
    # prelocate to store the sensitivity matrices
    dPjk = np.zeros([len(nodeLineNames), len(regs)])
    dV = np.zeros([len(nodeNames), len(regs)])
    # main loop through all regs
    for r, reg in enumerate(regs):
        # fresh compilation: to remove previous modifications
        dss.text(f"Compile [{dss_file}]")
        set_baseline(dss)
        # create a sensitivity object
        sen_obj = sensitivityPy(dss, time=0)
        # Perturb DSS with tap change
        sen_obj.perturbRegDSS(reg, 1.0 + 0.00625)  # +1 tap
        # solve dss file
        dss.text("solve")
        # compute Voltage sensitivity
        currVolts = sen_obj.voltageMags()
        dV[:, r] = currVolts - baseVolts
        # compute PTDF
        currPjk, _, _, _ = sen_obj.flows(nodeLineNames)
        dPjk[:, r] = currPjk - basePjk

    # save
    dfV = pd.DataFrame(dV, np.asarray(nodeNames), np.asarray(regs))
    dfV.to_pickle(pathlib.Path(script_path).joinpath("inputs", case, "VoltageToRegSensitivity.pkl"))
    dfPjk = pd.DataFrame(dPjk, np.asarray(nodeLineNames), np.asarray(regs))
    dfPjk.to_pickle(pathlib.Path(script_path).joinpath("inputs", case, "FlowsToRegSensitivity.pkl"))

    if initParams["plot"] == "True":
        h = 20
        w = 20
        ext = '.png'
        # VoltageSensitivity
        plt.clf()
        fig, ax = plt.subplots(figsize=(h, w))
        ax = sns.heatmap(dfV, annot=False)
        fig.tight_layout()
        output_img = pathlib.Path(script_path).joinpath("outputs", "VoltageToRegSensitivity" + ext)
        plt.savefig(output_img)
        plt.close('all')
        # Flows sensitivity
        plt.clf()
        fig, ax = plt.subplots(figsize=(h, w))
        ax = sns.heatmap(dfPjk, annot=False)
        fig.tight_layout()
        output_img = pathlib.Path(script_path).joinpath("outputs", "FlowsToRegSensitivity" + ext)
        plt.savefig(output_img)
        plt.close('all')
