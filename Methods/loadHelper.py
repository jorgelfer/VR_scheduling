# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:48:37 2022

@author: tefav
"""

import pandas as pd
import numpy as np

class loadHelper:
    
    def __init__(self, initfreq, finalFreq, price=False):
        
        self.initFreq = initfreq
        self.finalFreq = finalFreq
        self.price = price

    def process_pdFrame(self, df):
        
        # reset index and columns names 
        df.index, df.columns = [range(df.index.size), range(df.columns.size)]
        df = df.astype({0: 'float64'})

        # create index date range
        time = pd.date_range("00:00", "23:00", freq=self.initFreq)
        # transform string index into datetime index
        df.index = pd.to_datetime(time)
        
        if self.finalFreq == 'H':
            return df
        
        # interpolate
        df = self.__interpolate(df)
           
        # extrapolate
        df = self.__extrapolateDataframe(df)
        
        return df
    
    def process_pdSeries(self, pdSeries):
        
        # turn the series into dataframe
        df = pdSeries.to_frame()
        # reset index and columns names 
        df.index, df.columns = [range(df.index.size), range(df.columns.size)]
        df = df.astype({0: 'float64'})

        # create index date range
        time = pd.date_range("00:00", "23:00", freq=self.initFreq)
        # transform string index into datetime index
        df.index = pd.to_datetime(time)
        
        if self.finalFreq == 'H':
            return df
        
        # interpolate
        df = self.__interpolate(df)
           
        # extrapolate
        df = self.__extrapolate(df)
        
        return df
        
        
    def __interpolate(self, df):
                        
        # resample data in original frequency by new frequency
        if self.price:
            df = df.resample(self.finalFreq).bfill()
        else:
            #df = df.resample(self.finalFreq).bfill()
            df = df.resample(self.finalFreq).mean()
            df = df.interpolate(method='polynomial', order=3)
            
        return df
    
    
    def __extrapolate(self, df):

        # number of times freq in an hour
        # extract from freq
        f = int(''.join(filter(lambda i: i.isdigit(), self.finalFreq)))
        extend = int(60 / f)
        
        # Extrapolate the index first based on original index
        df = pd.DataFrame(
            data=df,
            index=pd.date_range(
                start=df.index[0],
                periods=len(df.index) + extend - 1,
                freq=df.index.freq
            )
        )
        
        # for prices back fill       
        if self.price:
            df.loc[df.index[-1],:] = df.loc[df.index[0],:] 
            df = df.resample(self.finalFreq).bfill()
            return df
        
        # for everythin else, polinomial interpolation
        # df.loc[df.index[-1],:] = df.loc[df.index[0],:] 
        # df = df.bfill()
        kw = dict(method='polynomial', order=3, fill_value="extrapolate", limit_direction="both")
        df = df.interpolate(**kw)

        return df

        
    def __extrapolateDataframe(self, df):

        # number of times freq in an hour
        # extract from freq
        f = int(''.join(filter(lambda i: i.isdigit(), self.finalFreq)))
        extend = int(60 / f)
        
        # matrix helper
        # mat = self.__dirichletMatrix(extend, df)
        # mat_aux = np.expand_dims(mat[-1].copy(), axis=0)
        # mat = np.concatenate([mat_aux, mat], axis=0)
        # mat = np.delete(mat, -1,0)
        
        # Extrapolate the index first based on original index
        df = pd.DataFrame(
            data=df,
            index=pd.date_range(
                start=df.index[0],
                periods=len(df.index) + extend - 1,
                freq=df.index.freq
            )
        )
    
        # Expected usage
        # df.loc[df.index[-1],:] = df.loc[df.index[0],:] 
        # df = df.resample(self.finalFreq).bfill()
        
        # df = df * mat
        
        kw = dict(method='polynomial', order=3, fill_value="extrapolate", limit_direction="both")
        df = df.interpolate(**kw)
        
        return df
    
        
    def __dirichlet(self, n):
        return np.random.dirichlet(np.ones(n))
    
    def __dirichletVector(self, extend):
        # create a stack
        stack = np.stack([self.__dirichlet(extend) for x in range(24)], axis=1)
        return stack.ravel(order='F')
    
    def __dirichletMatrix(self, extend, df):
        # create a stack
        stack = np.stack([self.__dirichletVector(extend) for x in range(len(df.columns))], axis=1) 
        return stack
