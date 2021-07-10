# encoding: utf-8
from utility import *
import math

#Error Vector Magnitude RMS(PE)/RMS(PR) only consider bpsk
def errorVecMag(file_name,phy_frames,n=48*8):
    if '+' in file_name:
        file_name=file_name.split('+')[0]+"/after_equalizer_symbols"+file_name.split('+')[-1]
    else:
        file_name="after_equalizer_symbols"+file_name
    for phy_frame in phy_frames:
        index=phy_frame.modules_index['after_equalizer']*48
        #鍑洪敊鐨勮瘽灏辫璁鸿瘉涓€涓媠ync_long鍜宖ft涔嬪悗鍑烘潵鐨勯噰鏍蜂釜鏁颁竴涓嶄竴鏍蜂簡
        complex_data=r_complexsamples(file_name,index,n)
        if len(complex_data)==0: 
            print('WTF')
            print('evm: '+str(index)+' wrong')
            exit()	
        #bpsk
        if phy_frame.feature['encoding']==0 or phy_frame.feature['encoding']==1:
            PE,PR=cal_PE_PR(complex_data,"bpsk",index)

        #qpsk
        if phy_frame.feature['encoding']==2 or phy_frame.feature['encoding']==3:
            PE,PR=cal_PE_PR(complex_data,"qpsk",index)

        #16qam
        if phy_frame.feature['encoding']==4 or phy_frame.feature['encoding']==5:
            PE,PR=cal_PE_PR(complex_data,"qam16",index)

        
        #64qam
        if phy_frame.feature['encoding']==6 or phy_frame.feature['encoding']==7:
            PE,PR=cal_PE_PR(complex_data,"qam64",index)
        
        
        EVM=((PE/n)**0.5)/((PR/n)**0.5)
        phy_frame.feature['evm']=EVM


    return phy_frames


def cal_PE_PR(complex_data,mod,index):
    PE,PR=0,0
    tor=abs(modulation[mod][0]-modulation[mod][1])/2
    for data in complex_data:
        distance=[tor,-1]  #[distance,index]
        for index,m_point in enumerate(modulation[mod]):
            temp=abs(data-m_point)
            if temp<distance[0]:
                distance[0],distance[1]=temp,index
        if distance[1]==-1:
            continue
        PE+=distance[0]**2 
        PR+=abs(modulation[mod][distance[1]])**2

    #debug
    if PR==0:
        print("wrong for "+mod+' '+str(index))
        for data in complex_data:
            distance=[tor,-1]  #[distance,index]
            for index,m_point in enumerate(modulation[mod]):
                temp=abs(data-m_point)
                if temp<distance[0]:
                    distance[0],distance[1]=temp,index
            if distance[1]==-1:
                #print('not match '+str(data))
                continue
            PE+=distance[0]**2 
            PR+=abs(modulation[mod][distance[1]])**2
        PE=0
        PR=1


    return PE,PR

bpsk=(complex(1,0),complex(-1,0))
qpsk=(complex(1/math.sqrt(2),1/math.sqrt(2)),complex(1/math.sqrt(2),-1/math.sqrt(2)),complex(-1/math.sqrt(2),-1/math.sqrt(2)),complex(-1/math.sqrt(2),1/math.sqrt(2)))
A=1/math.sqrt(10)
qam16=(complex(3*A,3*A),complex(3*A,A),complex(A,A),complex(A,3*A),
       complex(3*A,-3*A),complex(3*A,-A),complex(A,-A),complex(A,-3*A),
       complex(-3*A,-3*A),complex(-3*A,-A),complex(-A,-A),complex(-A,-3*A),
       complex(-3*A,3*A),complex(-3*A,A),complex(-A,A),complex(-A,3*A))
A=1/math.sqrt(4)
qam64=(complex(7*A,7*A),complex(7*A,5*A),complex(7*A,3*A),complex(7*A,A),
       complex(5*A,7*A),complex(5*A,5*A),complex(5*A,3*A),complex(5*A,A),
       complex(3*A,7*A),complex(3*A,5*A),complex(3*A,3*A),complex(3*A,A),
       complex(A,7*A),complex(A,5*A),complex(A,3*A),complex(A,A),

       complex(7*A,-7*A),complex(7*A,-5*A),complex(7*A,-3*A),complex(7*A,-A),
       complex(5*A,-7*A),complex(5*A,-5*A),complex(5*A,-3*A),complex(5*A,-A),
       complex(3*A,-7*A),complex(3*A,-5*A),complex(3*A,-3*A),complex(3*A,-A),
       complex(A,-7*A),complex(A,-5*A),complex(A,-3*A),complex(A,-A),

       complex(-7*A,-7*A),complex(-7*A,-5*A),complex(-7*A,-3*A),complex(-7*A,-A),
       complex(-5*A,-7*A),complex(-5*A,-5*A),complex(-5*A,-3*A),complex(-5*A,-A),
       complex(-3*A,-7*A),complex(-3*A,-5*A),complex(-3*A,-3*A),complex(-3*A,-A),
       complex(-A,-7*A),complex(-A,-5*A),complex(-A,-3*A),complex(-A,-A),

       complex(-7*A,7*A),complex(-7*A,5*A),complex(-7*A,3*A),complex(-7*A,A),
       complex(-5*A,7*A),complex(-5*A,5*A),complex(-5*A,3*A),complex(-5*A,A),
       complex(-3*A,7*A),complex(-3*A,5*A),complex(-3*A,3*A),complex(-3*A,A),
       complex(-A,7*A),complex(-A,5*A),complex(-A,3*A),complex(-A,A))


modulation={"bpsk":bpsk,"qpsk":qpsk,"qam16":qam16,"qam64":qam64}
