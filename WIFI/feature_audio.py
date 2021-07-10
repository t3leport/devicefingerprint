# -*- coding: utf-8 -*-
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import matplotlib.pyplot as plt
from utility import *
import numpy as np
from scipy import stats
from math import *

short_len,long_len,sig_data=128,128,0
Fs=20e6
TotalLen=short_len+long_len+sig_data

#从ltf和data 里取特征
def audio(file_name,phy_frames,total_len=TotalLen): 
    if '+' in file_name:
        file_name=file_name.split('+')[0]+"/after_sync_short"+file_name.split('+')[-1]
    else:
        file_name="after_sync_short"+file_name
    for phy_frame in phy_frames:
        index=phy_frame.modules_index['after_sync_short']
        #只取long_len+data_len
        complex_data=r_complexsamples(file_name,index,total_len)
        if len(complex_data)==0:
            print('WTF_stats')
            print('read from '+file_name+' index: '+str(index)+' out of range')
            exit()  
        
        complex_data_t=[]
        
        for i,data in enumerate(complex_data):
            data_in_tdom=(data*e**complex(0,2*pi*phy_frame.nomfreq/Fs)).real
            #print(data_in_tdom)
            complex_data_t.append(data_in_tdom)

        complex_data_t=np.array(complex_data_t)
        F,f_names=ShortTermFeatures.feature_extraction(complex_data_t,Fs,256,256)
        phy_frame.feature['zcr']=F[0][0]
        #print(len(F[0]))
        #phy_frame.feature['energy_entropy']=F[2][1]
        phy_frame.feature['spectral_centroid']=F[3][0]
        phy_frame.feature['spectral_spread']=F[4][0]
        #phy_frame.feature['spectral_entropy']=F[5][1]
        #phy_frame.feature['spectral_flux']=F[6][1]
       

        F,f_names=ShortTermFeatures.feature_extraction(complex_data_t,Fs,128,128)
        phy_frame.feature['spectral_flux_long']=F[6][1]
        phy_frame.feature['energy_entropy_long']=F[2][1]
        phy_frame.feature['spectral_entropy_long']=F[5][1]

        phy_frame.feature['energy_entropy_short']=F[2][0]
        phy_frame.feature['spectral_entropy_short']=F[5][0]
        
        if F[0][0]==0:
            F[0][0]=1e-6
        phy_frame.feature['zcr_var']=F[0][1]/F[0][0]
        
        phy_frame.feature['energy_entropy_var']=F[2][1]/F[2][0]


        #static_feature
        #print("e "+str(len(F[1])))
        F,f_names=ShortTermFeatures.feature_extraction(complex_data_t,Fs,64,64)
        #print(F[6])
        phy_frame.feature['energy_var']=np.var(F[1])
        #phy_frame.feature['zcr_var']=np.var(F[0])
        phy_frame.feature['spectral_entropy_var']=np.var(F[5])
        phy_frame.feature['spectral_flux_var']=np.var(F[6][1:])
        #phy_frame.feature['energy_entropy_var']=np.var(F[2])
        phy_frame.feature['spectral_centroid_var']=np.var(F[3])
        phy_frame.feature['spectral_spread_var']=np.var(F[4])
        '''
        phy_frame.feature['zcr_var']=np.var(F[0])
        phy_frame.feature['energy_var']=F[1][1]/F[1][0]
        phy_frame.feature['spectral_entropy_var']=F[5][1]/F[5][0]
        phy_frame.feature['energy_entropy_var']=F[2][1]/F[2][0]
        '''


    return phy_frames


