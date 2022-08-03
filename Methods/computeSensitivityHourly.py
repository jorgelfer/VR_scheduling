# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:00:01 2022

@author: tefav
"""

from Methods.sensitivityPy import sensitivityPy 
import numpy as np
import pandas as pd
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt


def disableGen(dss, gen):
    dss.circuit_set_active_element('generator.' + gen)
    #debug
    # a = dss.cktelement_read_enabled() == 1
    dss.text(f"disable generator.{gen}")

def computeSensitivity(script_path, case, dss, plot, ite):

    # create a sensitivity object
    sen_obj = sensitivityPy(dss, time=0)
    
    # get all node-based base volts 
    nodeBaseVoltage = sen_obj.get_nodeBaseVolts()

    # get all node-based buses, 
    nodeNames = dss.circuit_all_node_names()

    # get base voltage
    baseVolts = sen_obj.voltageProfile()
    
    # prelocate to store the sensitivity matrices
    dvdp = np.zeros([len(nodeNames),len(nodeNames)]) # containing volts 
    dvdq = np.zeros([len(nodeNames),len(nodeNames)]) # containing volts 
    
    # main loop through all nodes 
    for n, node in enumerate(nodeNames):

        # create a sensitivity object
        sen_obj = sensitivityPy(dss, time=0)

        # Perturb DSS with small gen (P) 
        sen_obj.perturbDSS(node, kv=nodeBaseVoltage[node], kw=10, P=True) # 10 kw
        
        dss.text("solve")

        # compute Voltage sensitivity
        currVolts = sen_obj.voltageProfile()
        dvdp[:,n] =  currVolts- baseVolts

        # Perturb DSS with small gen (Q) 
        sen_obj.perturbDSS(node, kv=nodeBaseVoltage[node], kw=10, P=False) # 10 kw
        
        dss.text("solve")

        # compute Voltage sensitivity
        currVolts = sen_obj.voltageProfile()
        dvdq[:,n] =  currVolts- baseVolts
        
        # disable Generator
        name = node.replace(".","_")
        disableGen(dss, name)
        
    # save
    dvdp = pd.DataFrame(dvdp, np.asarray(nodeNames), np.asarray(nodeNames)) / 10
    dvdq = pd.DataFrame(dvdq, np.asarray(nodeNames), np.asarray(nodeNames)) / 10

    if plot:
        h = 20
        w = 20
        ext = '.png'
        
        # VoltageSensitivity
        plt.clf()
        fig, ax = plt.subplots(figsize=(h,w))                
        sns.heatmap(dvdp, annot=False)
        fig.tight_layout()
        output_img = pathlib.Path(script_path).joinpath("outputs", f"VoltageSensitivity_{ite}" + ext)
        plt.savefig(output_img)
        plt.close('all')
        
    return dvdp, dvdq 
