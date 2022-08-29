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
    dss.text(f"Compile [{dss_file}]")  # fresh compilation
    set_baseline(dss)
    # create a sensitivity object
    sen_obj = sensitivityPy(dss, time=0)
    # get all node-based base volts
    nodeBaseVoltage = sen_obj.get_nodeBaseVolts()
    # get all node-based buses
    nodeNames = dss.circuit_all_node_names()
    # get all node-based lines names
    nodeLineNames, lines = sen_obj.get_nodeLineNames()
    # list DSS regulators
    trafos = dss.transformers_all_Names()
    regs = [tr for tr in trafos if "reg" in tr]
    # prelocate to store the sensitivity matrices
    dPjkL = np.zeros([len(nodeLineNames), len(regs)])
    dPjkR = np.zeros([len(nodeLineNames), len(regs)])
    dVL = np.zeros([len(nodeNames), len(regs)])
    dVR = np.zeros([len(nodeNames), len(regs)])
    # main loop through all regs
    for r, reg in enumerate(regs):
        # Excecution from init value
        # fresh compilation: to remove previous modifications
        dss.text(f"Compile [{dss_file}]")
        set_baseline(dss)
        # create a sensitivity object
        sen_obj = sensitivityPy(dss, time=0)
        # solve dss file
        dss.text("solve")
        # get base voltage
        baseVolts = sen_obj.voltageMags()
        # get base pjk
        basePjk, _, _, _ = sen_obj.flows(nodeLineNames)

        # Excecution from final value
        dss.text(f"Compile [{dss_file}]")
        set_baseline(dss)
        # create a sensitivity object
        sen_objR = sensitivityPy(dss, time=0)
        # Perturb DSS with tap change
        sen_objR.perturbRegDSS(reg, 1.0 + 0.00625 * (8))  # +1 tap
        # solve dss file
        dss.text("solve")
        # compute R sensitivity
        VR = sen_objR.voltageMags()
        dVR[:, r] = (VR - baseVolts) / 8
        # compute PTDF
        PjkR, _, _, _ = sen_objR.flows(nodeLineNames)
        dPjkR[:, r] = (PjkR - basePjk)  / 8
        

        # Excecution from init value
        dss.text(f"Compile [{dss_file}]")
        set_baseline(dss)
        # create a sensitivity object
        sen_objL = sensitivityPy(dss, time=0)
        # Perturb DSS with tap change
        sen_objL.perturbRegDSS(reg, 1.0 + 0.00625 * (-8))  # +1 tap
        # solve dss file
        dss.text("solve")
        # compute L sensitivity
        VL = sen_objL.voltageMags()
        dVL[:, r] = (baseVolts - VL) / 8
        # compute PTDF
        PjkL, _, _, _ = sen_obj.flows(nodeLineNames)
        dPjkL[:, r] = (basePjk - PjkL) / 8
    # save
    dfPjkR = pd.DataFrame(dPjkR, np.asarray(nodeLineNames), np.asarray(regs))
    dfPjkR.to_pickle(pathlib.Path(script_path).joinpath("inputs", case, "FlowsToRegSensitivityR.pkl"))
    dfPjkL = pd.DataFrame(dPjkL, np.asarray(nodeLineNames), np.asarray(regs))
    dfPjkL.to_pickle(pathlib.Path(script_path).joinpath("inputs", case, "FlowsToRegSensitivityL.pkl"))


    # save volts
    dfVR = pd.DataFrame(dVR, np.asarray(nodeNames), np.asarray(regs))
    dfVR.to_pickle(pathlib.Path(script_path).joinpath("inputs", case, "VoltageToRegSensitivityR.pkl"))
    dfVL = pd.DataFrame(dVL, np.asarray(nodeNames), np.asarray(regs))
    dfVL.to_pickle(pathlib.Path(script_path).joinpath("inputs", case, "VoltageToRegSensitivityL.pkl"))

    if initParams["plot"] == "True":
        h = 20
        w = 20
        ext = '.png'
        # Flows sensitivity R
        plt.clf()
        fig, ax = plt.subplots(figsize=(h, w))
        ax = sns.heatmap(dfPjkR, annot=False)
        fig.tight_layout()
        output_img = pathlib.Path(script_path).joinpath("outputs", "FlowsToRegSensitivityR" + ext)
        plt.savefig(output_img)
        plt.close('all')
        # Flows sensitivity L
        plt.clf()
        fig, ax = plt.subplots(figsize=(h, w))
        ax = sns.heatmap(dfPjkL, annot=False)
        fig.tight_layout()
        output_img = pathlib.Path(script_path).joinpath("outputs", "FlowsToRegSensitivityL" + ext)
        plt.savefig(output_img)
        plt.close('all')
        # volts sensitivity R
        plt.clf()
        fig, ax = plt.subplots(figsize=(h, w))
        ax = sns.heatmap(dfVR, annot=False)
        fig.tight_layout()
        output_img = pathlib.Path(script_path).joinpath("outputs", "VoltageToRegSensitivityR" + ext)
        plt.savefig(output_img)
        plt.close('all')
        # volts sensitivity L
        plt.clf()
        fig, ax = plt.subplots(figsize=(h, w))
        ax = sns.heatmap(dfVL, annot=False)
        fig.tight_layout()
        output_img = pathlib.Path(script_path).joinpath("outputs", "VoltageToRegSensitivityL" + ext)
        plt.savefig(output_img)
        plt.close('all')
