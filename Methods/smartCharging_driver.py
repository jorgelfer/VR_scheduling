from Methods.SmartCharging import SmartCharging 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib


class smartCharging_driver:

    def __init__(self, ext, arrivalTime_list, departureTime_list, initEnergy_list, evCapacity_list, initW_list):
        # constructor
        self.arrivalTime_list = arrivalTime_list
        self.departureTime_list = departureTime_list
        self.initEnergy_list = initEnergy_list
        self.evCapacity_list = evCapacity_list
        self.initW_list = initW_list
        
        self.ext = ext
        

    def charging_driver(self, output_dir, it, demandProfile, LMP, LMP_index, plot):
        
        # preprocess
        self.output_dir = output_dir
        self.it = it
        
        # compute total demand base case
        totalDemand =  demandProfile.sum(axis = 0).to_frame()
    
        # prelocate EV demand
        EV_demandProfile = np.zeros(demandProfile.shape)
        EV_demandProfile = pd.DataFrame(EV_demandProfile, index=demandProfile.index, columns=demandProfile.columns)
        
        # create smart charging object
        charging_obj = SmartCharging(numberOfHours=24, pointsInTime=LMP.shape[1]) 
        
        for i, ind in enumerate(LMP_index):
            #individual LMP (pi)
            pi = LMP.loc[ind,:]
            # reorder dates
            pi = self.__reorder_dates(pi)
            # transform to array
            pi = np.expand_dims(pi.values, axis=1)
            # household demand profile
            PH = demandProfile.loc[ind,:] # normal demand
            # reorder dates
            PH = self.__reorder_dates(PH)
            PH = np.expand_dims(PH.values, axis=1)
            # user defined weights
            w = np.squeeze(self.initW_list[i]) 
            #EV initial conditions
            arrTime = self.arrivalTime_list[i]
            depTime = self.departureTime_list[i]
            initEnergy = self.initEnergy_list[i]
            evCapacity = self.evCapacity_list[i] 
        
            # optimal EV charging using the smart charging object
            PV_star,_,_,_ = charging_obj.QP_charging(pi, PH, w, arrTime, depTime, initEnergy, evCapacity) # user specific values
            # reorder index from dataSeries:
            PV_star = self.__order_dates(PV_star, freq="30min")
            # assign to the profile
            EV_demandProfile.loc[ind,:] = PV_star
            
        #define new demand
        newDemand = demandProfile.values + EV_demandProfile.values
        newDemand = pd.DataFrame(newDemand, index=demandProfile.index, columns=demandProfile.columns)
    
        if plot:
            # new demand plot
            plt.clf()
            fig, ax = plt.subplots()
            
            totalNewDemand = newDemand.sum(axis = 0).to_frame()
            concat3 = pd.concat([totalDemand, totalNewDemand], axis=1)
            concat3.plot()
            plt.legend(['load_toalDemand', 'load_EV_totalDemand'], prop={'size': 10})
                
            fig.tight_layout()
            output_img = pathlib.Path(self.output_dir).joinpath(f"EVcorrected_demand_{self.it}"+ self.ext)
            plt.savefig(output_img)
            plt.close('all')

        return newDemand
    
    def __reorder_dates(self, pdSeries):
        pdSeries1 = pdSeries[:pdSeries.index.get_loc('08:00')]
        pdSeries2 = pdSeries[pdSeries.index.get_loc('08:00'):]
        pdSeries = pd.concat((pdSeries2, pdSeries1))
        return pdSeries
    
    def __order_dates(self, array, freq="30min"):
        # initialize time pdSeries
        pdSeries = pd.Series(np.zeros(len(array)))
        pdSeries.index = pd.date_range("00:00", "23:59", freq=freq).strftime('%H:%M')
        # reorder init dataSeries to match Kartik's order
        pdSeries = self.__reorder_dates(pdSeries)
        # assign obtained values
        pdframe = pdSeries.to_frame()
        pdframe[0] = array
        pdSeries = pdframe.squeeze()
        # order back to normal 
        pdSeries1 = pdSeries[:pdSeries.index.get_loc('00:00')]
        pdSeries2 = pdSeries[pdSeries.index.get_loc('00:00'):]
        pdSeries = pd.concat((pdSeries2, pdSeries1))
        return pdSeries

