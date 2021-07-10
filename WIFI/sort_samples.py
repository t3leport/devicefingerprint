# encoding: utf-8
import matplotlib.pyplot as plt
#from pylab import * 
import optparse
#import numpy as np
#import seaborn as sns
import re
import os
#from prettyprinter import cpprint
import math
import numpy as np
import feature_evm as f_evm
import feature_iq_offset as f_iq
import feature_stats as f_stats
import feature_audio as f_audio
from utility import *

'''
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
'''
mac_iot=['04:CF:8C:F4:EB:C7','04:CF:8C:A0:FD:23','90:97:D5:32:F9:8A','04:CF:8C:AD:33:95',
        '0C:9D:92:4F:3F:58','7C:49:EB:18:F3:7C','40:31:3C:BB:C6:94','50:EC:50:02:25:A3',
        '44:23:7C:D5:24:B4','04:CF:8C:02:49:22','28:6D:CD:01:FB:97','6C:21:A2:C7:87:C1',
        '10:D0:7A:39:1B:1A','30:6A:85:C9:F0:3B','5C:1D:D9:5E:FF:A1','E4:95:6E:45:10:88','3C:22:FB:80:39:FE',
        '04:CF:8C:B6:F6:6A','04:CF:8C:B6:F6:84','04:CF:8C:B4:C9:5B']



#statistic feature
def main():
    parser=optparse.OptionParser("'usage -f <log file>,  -s <choose sample file >")
    parser.add_option('-f',dest='file',type='string',help='specify file within fre_offset')
    parser.add_option('-s',dest='filesink',type='string',help='choose sample file')
    parser.add_option('-n',dest='num_of_frame',type='int',help='the number of frames')
    (options,args)=parser.parse_args()
    if (options.file==None):
        print(parser.usage)
        exit(0)
    file=options.file
    f=open(file)
    phy_frames=collect_phy_frames(f)
    #plot_consetellation(phy_frames,macaddr)
    #file_name="after_sync_long_fft_11_20.bin"
    file_name=options.filesink
    num_of_frame=options.num_of_frame
    phy_frames=f_evm.errorVecMag(file_name,phy_frames,48*10) 
    phy_frames=f_iq.iqoffset(file_name,phy_frames,48*10)
    phy_frames=f_audio.audio(file_name,phy_frames)

    
    for phy_frame in phy_frames:
        print(phy_frame)
    
    #print(c_features)
    save_files(phy_frames,file_name,mac_iot,num_of_frame)
    print_mac_info(phy_frames,mac_pdt)
    #plot_consetellation("after_sync_long_fft11-23-19:36.bin", 657926784+2*64,48)
    #plot_consetellation("after_equalizer_symbols11-23-19:36.bin", 1540348*48,48)
    #same_feature_diff_macs(phy_frames,"zcr",["E4:95:6E:45:10:88","40:31:3C:BB:C6:94"],n=100)
    #feature_boxplot(phy_frames,['freqofs','evm','iq_offset','snr'],n=100)




if __name__ == '__main__':
    main()
