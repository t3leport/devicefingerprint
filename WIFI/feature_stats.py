# encoding: utf-8
from utility import *
import numpy as np
from scipy import stats
from math import *

short_len,long_len=128,128
Ts=20e6

#从stf和ltf里取统计特征
def stat(file_name,phy_frames,n=64*4): 
    file_name="after_sync_short"+file_name
    for phy_frame in phy_frames:
        index=phy_frame.modules_index['after_sync_short']
        #鍑洪敊鐨勮瘽灏辫璁鸿瘉涓€涓媠ync_long鍜宖ft涔嬪悗鍑烘潵鐨勯噰鏍蜂釜鏁颁竴涓嶄竴鏍蜂簡
        complex_data=r_complexsamples(file_name,index,n)
        if len(complex_data)==0:
            print('WTF_stats')
            print('read from '+file_name+' index: '+str(index)+' out of range')
            exit()	
        print('fuck')
        
        complex_data_t=[]
        
        for i,data in enumerate(complex_data):
            data_in_tdom=(data*e**complex(0,2*pi*phy_frame.nomfreq/Ts)).real
            #print(data_in_tdom)
            complex_data_t.append(data_in_tdom)
    
        print(complex_data_t)
        #complex_data_t=complex_data

        win_size_short=16
        win_size_long=64
        win_num_short=int(short_len/win_size_short)
        win_num_long=int(long_len/win_size_long)
        f_stats=[]

        #sync_short
        for i in range(win_num_short):
            temp=complex_data_t[i*16:(i+1)*16]
            append_awin(f_stats,temp)

        #sync_long
        for i in range(win_num_long):
            temp=complex_data_t[i*64:(i+1)*64]
            append_awin(f_stats,temp)
        
        phy_frame.feature['stats']=f_stats
    return phy_frames


def append_awin(f_stats,temp):
    f_stats.append(np.var(temp))
    f_stats.append(stats.skew(np.array(temp)))
    f_stats.append(stats.kurtosis(np.array(temp)))
    f_stats.append(np.median(temp))
