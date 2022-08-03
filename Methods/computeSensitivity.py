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
    dss.text("Set controlmode=Off") # this is for disabling regulators

def computeSensitivity(script_path, case, dss, dss_file, plot):

    set_baseline(dss)

    # create a sensitivity object
    sen_obj = sensitivityPy(dss, time=0)
    
    # get all node-based base volts 
    nodeBaseVoltage = sen_obj.get_nodeBaseVolts()

    # get all node-based buses, 
    nodeNames = dss.circuit_all_node_names()

    # get all node-based lines names
    nodeLineNames, lines = sen_obj.get_nodeLineNames()

    dss.text("solve")

    # get base voltage
    baseVolts, _ = sen_obj.voltageProfile()
    
    # get base pjk 
    basePjk,_,_,_ = sen_obj.flows(nodeLineNames)
    
    # prelocate to store the sensitivity matrices
    PTDF_jk = np.zeros([len(nodeLineNames),len(nodeNames)]) # containing flows Pjk
    VS = np.zeros([len(nodeNames),len(nodeNames)]) # containing volts 
    
    # main loop through all nodes 
    for n, node in enumerate(nodeNames):
        # fresh compilation to remove previous modifications
        dss.text(f"Compile [{dss_file}]")
        set_baseline(dss)

        # create a sensitivity object
        sen_obj = sensitivityPy(dss, time=0)

        # Perturb DSS with small gen 
        sen_obj.perturbDSS(node, kv=nodeBaseVoltage[node], kw=10, P=True) # 10 kw
        
        dss.text("solve")

        # compute Voltage sensitivity
        currVolts, _ = sen_obj.voltageProfile()
        VS[:,n] =  currVolts- baseVolts
        
        # compute PTDF
        currPjk, _, _, _ = sen_obj.flows(nodeLineNames)
        PTDF_jk[:,n] =  currPjk - basePjk

    # save
    dfVS = pd.DataFrame(VS, np.asarray(nodeNames), np.asarray(nodeNames))
    dfVS.to_pickle(pathlib.Path(script_path).joinpath("inputs", case,"VoltageSensitivity.pkl"))
    dfPjk = pd.DataFrame(PTDF_jk,np.asarray(nodeLineNames), np.asarray(nodeNames))
    dfPjk.to_pickle(pathlib.Path(script_path).joinpath("inputs",case, "PTDF_jk.pkl"))

    if plot:
        h = 20
        w = 20
        ext = '.png'
        
        # VoltageSensitivity
        plt.clf()
        fig, ax = plt.subplots(figsize=(h,w))                
        ax = sns.heatmap(dfVS, annot=False)
        fig.tight_layout()
        output_img = pathlib.Path(script_path).joinpath("outputs","VoltageSensitivity" + ext)
        plt.savefig(output_img)
        plt.close('all')
        
        # PTDF
        plt.clf()
        fig, ax = plt.subplots(figsize=(h,w))                
        ax = sns.heatmap(dfPjk, annot=False)
        fig.tight_layout()
        output_img = pathlib.Path(script_path).joinpath("outputs","PTDF" + ext)
        plt.savefig(output_img)
        plt.close('all')
