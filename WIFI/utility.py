# encoding: utf-8
import struct
import numpy as np
import os,re
import matplotlib.pyplot as plt
import matplotlib
import itertools

def r_complexsamples(filename,star,length):
    data_raw=[]
    star=int(star)
    length=int(length)
    f=open(filename,mode='rb')
    real=1 #1娴狅綀銆冪€圭偤鍎撮敍锟�0娴狅綀銆冮搹姘跺劥
    realpart=0
    imgpart=0
    complex_data=[]
    i=0
    f.seek(8*star,0)
    while i<length:
        byte=f.read(4*1)
        if not byte:break
        afloat=struct.unpack('f'*1,byte)
        data_raw.append(afloat)
        if real==1:
            realpart=afloat
            real=0
        else:
            imgpart=afloat
            #print(complex(realpart[0],imgpart[0]))
            complex_data.append(complex(realpart[0],imgpart[0]))
            real=1
            i+=1
    f.close()
    return complex_data


def collect_phy_frames(f):
    """
    return phy_frames[frame1,frame2]
    """
    phy_frames=[]
    for line in f.readlines():
        if "type" not in line:
            continue
        mac_addr,ftype,subtype,nomfreq,modules_index,feature="","","",0.0,dict(),dict()
        p1 = re.compile(r'[(](.*?)[)]', re.S)
        aline=re.findall(p1, line[1:-1])
        #print(aline)
        for i in aline:
            i.strip('\'')
            key,val=[j.strip(' ') for j in i.split('.',1)]

            if key=="type":
                ftype=val
            elif key=="mac_addr":
                mac_addr=val.upper()
            elif key=="subtype":
                subtype=val
            elif key=="nomfreq":
                nomfreq=float(val)
            elif key=="freqofs":
                feature['freqofs']=float(val)
            elif key=="snr":
                feature['snr']=float(val)
            elif key=="encoding":
                feature['encoding']=int(val)
            elif key=="after_sync_short":
                modules_index['after_sync_short']=int(val)
            elif key=="after_sync_long":
                modules_index['after_sync_long']=int(val)
            elif key=="after_equalizer":
                modules_index['after_equalizer']=int(val)

        #print(phy_frame(mac_addr,ftype,subtype,nomfreq,freqofs,snr,encoding,modules_index))
        phy_frames.append(phy_frame(mac_addr,ftype,subtype,nomfreq,modules_index,feature))

    return phy_frames

def save_files(phy_frames,file_name,mac_iot=[],min_numofsamples=100):
    if '+' not in file_name:
        path=os.path.join(os.getcwd(),file_name.split('.')[0])
    else:
        path=os.path.join(os.getcwd(),file_name.split('+')[-1].split('.')[0])
    print(path)
    if not os.path.exists(path):
    # 如果不存在则创建目录
    #创建目录操作函数
        os.makedirs(path)
    #os.chdir('./'+path)
    

    #004:CF:8C:F4:EB:C7  空气净化器
    specify_mac=0 if len(mac_iot)==0 else 1
    mac_count=dict()
    #超过100个帧的样本  macaddress:[phy_frame]
    mac_addrs=dict()
    #统计所有的mac
    for phy_frame in phy_frames:
        if specify_mac==1:
            if phy_frame.mac_addr not in mac_iot:
                continue
        if phy_frame.mac_addr not in mac_addrs:
            mac_addrs[phy_frame.mac_addr]=[phy_frame]
            mac_count[phy_frame.mac_addr]=1
        else:
            mac_addrs[phy_frame.mac_addr].append(phy_frame)
            mac_count[phy_frame.mac_addr]+=1
    #print(mac_count)
    #print(mac_iot)
    
    #样本数不超过min_numofsamples的从mac_addrs中移除
    for mac in list(mac_addrs.keys()):
        if mac_count[mac]<min_numofsamples:
            del(mac_addrs[mac])

    n_index = range(len(mac_addrs))
    str_index = [str(x) for x in n_index]
    mac_to_index = dict(zip([key for key in mac_addrs], str_index))
    

    for mac,mac_phy_frames in mac_addrs.items():
        print(mac)
        '''
        if mac_count[mac]<min_numofsamples:
            continue
        '''
        for count,phy_frame in enumerate(mac_phy_frames):
            
            mac_folder = os.path.join(path,mac)
            if not os.path.exists(mac_folder):
                # 如果不存在则创建目录
                #创建目录操作函数
                os.makedirs(mac_folder)
            
            count+=1
            
            #print(index_label)
            if count>min_numofsamples and min_numofsamples!=0:
                continue

            feature_dict=phy_frame.feature
            header_sample=[feature_dict[key] for key in feature_order]
            #header_sample=[feature_dict['freqofs'],feature_dict['encoding'],feature_dict['evm']]
            # map string to index
            index_label = mac_label[mac] if specify_mac==1 else mac_to_index[mac]
            header_sample.append(index_label)
            #print(header_sample)
            #print(str(mac_addr)[2:-1]+str(header_sample)+str(wlan))
            np.save(os.path.join(mac_folder, str(count)), np.array(header_sample))
    
    #print the encoding of label
    for mac,mac_phy_frames in mac_addrs.items():
        index_label = mac_iot.index(mac) if specify_mac==1 else mac_to_index[mac]
        header_sample.append(index_label)
        print("mac:{},label:{}".format(mac,index_label))

    print('saved!')


#features order, freqofs,encoding,evm,iq_offset
def load_phyframe(pathname,mac_iot,c_features):
    phy_frames=[]
    X,Y=load_samples(pathname,c_features)
    for row,label in enumerate(Y):
        mac_addr=[k for k,v in mac_label.items() if v == label][0]
        feature=dict()
        for index,val in enumerate(X[row]):
            feature[c_features.split(',')[index]]=val
        phy_frames.append(phy_frame(mac_addr,"?","?","?",{"?":"?"},feature))
    
    return phy_frames

    
def plot_consetellation(file_name,index,length,marker="o",color='red'):
    complex_datas=r_complexsamples(file_name,index,length)
    for complex_data in complex_datas:
        #print(complex_data)
        plt.scatter(complex_data.real, complex_data.imag, marker = marker,color = color, s = 40 ,label = 'First')
    plt.show()


#continuing
def plot_consetellation_all(file_name,phy_frames,macaddr):    
    """
    file_name:str
    phy_frames:list
    macaddr:list
    """
    n=144 #length of signal,256for stf and ltf
    macaddr=[ i.upper() for i in macaddr.split(',')]
    for phy_frame in phy_frames:
        if phy_frame.mac_addr not in macaddr:
            continue
        index=phy_frame.modules_index['after_sync_long']
        #index=401203
        print(index*48)
        complex_datas=r_complexsamples(file_name,index*48,n)
        marker,color=f(macaddr.index(phy_frame.mac_addr)).split(',')
        for complex_data in complex_datas:
            print(complex_data)
            plt.scatter(complex_data.real, complex_data.imag, marker = marker,color = color, s = 40 ,label = 'First')
    plt.show()


def same_feature_diff_macs(phy_frames,feature_name,macaddr=[],n=100):
    """
    phy_frames:list
    """
    if len(macaddr)==0:
        print("pl specify some mac address")
        return
    macaddr=[ i.upper() for i in macaddr]
    mac_dict=dict()
    for phy_frame in phy_frames:
        if phy_frame.mac_addr not in macaddr:
            continue
        if phy_frame.mac_addr not in mac_dict:
            mac_dict[phy_frame.mac_addr]=[phy_frame.feature[feature_name]]
        else:
            if len(mac_dict[phy_frame.mac_addr])>100:
                continue
            mac_dict[phy_frame.mac_addr].append(phy_frame.feature[feature_name])
    
    for mac in mac_dict:
        print("plot "+mac)
        marker,color=f(macaddr.index(mac)).split(',')
        x=[i for i in range(len(mac_dict[mac]))]
        print("方差"+str(np.var(mac_dict[mac])))
        print("平均值"+str(np.mean(mac_dict[mac])))
        plt.plot(x, mac_dict[mac],marker = marker,color = color,label=mac)

    plt.title(feature_name.upper())
    plt.xlabel('x/sample')
    plt.ylabel('y/value')
    plt.legend()
    plt.show()
        

def feature_boxplot(phy_frames,feature_name,n=100):
    """
    phy_frames:list
    feature_name:list
    """
    '''
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    '''
    Title={"freqofs":"Frequency Offset","evm":"EVM","iq_offset":"I/Q Offset","zcr":"ZCR",'zcr_var':"ZCR_var",'energy_var':"energy_var",'energy_entropy':"Energy_entropy",'energy_entropy_var':'energy_entropy_var','spectral_centroid':"Spectral_centroid",'spectral_spread':"Spectral_spread",'spectral_entropy':"Spectral_entropy",'spectral_entropy_var':"spectral_entropy_var",'spectral_flux':"spectral_flux",'spectral_flux_var':"spectral_flux_var"}
    mac_dict=dict()
    fig=plt.figure(num=1,figsize=(4,4))
    pict_len=len(feature_name)

    for phy_frame in phy_frames:
        if phy_frame.mac_addr not in mac_pdt:
            continue
        if phy_frame.mac_addr not in mac_dict:
            mac_dict[phy_frame.mac_addr]=[phy_frame.feature]
        else:
            if len(mac_dict[phy_frame.mac_addr])>100:
                continue
            mac_dict[phy_frame.mac_addr].append(phy_frame.feature)

    for i,feature in enumerate(feature_name):
        #feature_array=[[],[]],  mac_dict={"mac":[{"snr":1,"freqofs":1},[]]}
        feature_mac_array=[]
        for mac,featuress in mac_dict.items():
            temp=[]
            for features in featuress:
                temp.append(features[feature])
            feature_mac_array.append(temp)
        #bar_labels = [mac_pdt[mac] for mac in mac_dict]
        bar_labels = [i for i in range(len(mac_dict))]
        print([mac_pdt[mac] for mac in mac_dict])
        fig.add_subplot(int("1"+str(pict_len)+str(i+1)))
        #plt.xticks([x+1 for x in range(len(feature_mac_array))], bar_labels)
        #plt.title(feature.upper())
        plt.title(Title[feature])
        bplt=plt.boxplot(feature_mac_array, notch=False, sym='2', vert=True, patch_artist=True)

        
        for pacthes, color in zip(bplt['boxes'], colors):
            pacthes.set_facecolor(color)

    plt.show()


def f(x):
    return {
        0: 'o,red',
        1: 'x,green',
        2: '*,blue',
        3: '+,yellow',
        4: '1,black',
        5: '3,purple',
        6: '^,tan',
        7: '>,darkkhaki',
        8: '2,aquamarine',
        9:'4,gainsboro',
        10:'h,pink',
        11: 'p,mediumaquamarine',
        12:'x,tomato',
        13:'D,lightblue',
        14:'|,wheat'
    }[x]


def print_mac_info(phy_frames,mac_pdt):
    mac_all=dict()
    for phy_frame in phy_frames:
        if phy_frame.mac_addr not in mac_all:
            mac_all[phy_frame.mac_addr]=1
        else:
            mac_all[phy_frame.mac_addr]+=1
    
    for mac in mac_all:
        if mac in mac_pdt:
            print("mac:{},count:{},product:{}".format(mac,mac_all[mac],mac_pdt[mac]))
        else:
            print("mac:{},count:{}".format(mac,mac_all[mac]))




def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if re.match(r'.*\d.*', f):
                fullname = os.path.join(root, f)
                yield fullname

def load_samples(pathname,c_features):
    """
    c_features:str format:"iq_offset,evm,freqofs"
    """
    prev_base=os.getcwd()
    os.chdir(pathname)
    base=os.getcwd()
    print(base)
    X=[]
    Y=[]
    for i in findAllFile(base):
        #print(i)
        vector=np.load(i).tolist()
        #print(vector)
        Y.append(int(vector.pop()))

        X.append(list(map(float,vector)))
        
    X=np.array(X)

    temp=[]
    for row in X:
        temp.append([list(row)[feature_order.index(key)] for key in c_features.split(',')])
    X=np.array(temp)

    Y=np.array(Y)
    os.chdir(prev_base)
    return X,Y


#https://blog.csdn.net/Tsehooo/article/details/109729814?utm_medium=distribute.pc_relevant.none-task-blog-baidulandingword-3&spm=1001.2101.3001.4242

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
 
    print(cm)
 
    
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
 
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.show()




colors = ['mediumaquamarine','tomato','pink', 'lightblue', 'lightgreen','red','darkkhaki','tan','aquamarine','gainsboro','indigo','orange','turquoise','wheat','seashell']

mac_pdt={'04:CF:8C:F4:EB:C7':u'空气净化器',
        '04:CF:8C:A0:FD:23':u'多功能网关',
        '90:97:D5:32:F9:8A':u'空气检测仪',
        '04:CF:8C:AD:33:95': u'智米加湿器',
        '0C:9D:92:4F:3F:58':u'ASUSTek路由器',
        '7C:49:EB:18:F3:7C':u'智能排插',
        '40:31:3C:BB:C6:94':u'万能遥控器',
        '50:EC:50:02:25:A3':u'扫地机器人',
        '44:23:7C:D5:24:B4':u'led灯泡',
        '3C:22:FB:80:39:FE':u'mbp',
        '28:6D:CD:01:FB:97':u'华为空气净化器',
        '6C:21:A2:C7:87:C1':u'音响',
        '10:D0:7A:39:1B:1A':u'投影仪',
        '30:6A:85:C9:F0:3B':u'三星平板',
        '5C:1D:D9:5E:FF:A1':u'iphone手机',
        'E4:95:6E:45:10:88':u'广联智通路由器',
        '04:CF:8C:02:49:22':u'摄像头',
        '04:CF:8C:B6:F6:6A':u'led灯泡3',
        '04:CF:8C:B6:F6:84':u'led灯泡2',
        '04:CF:8C:B4:C9:5B':u'led灯泡1',
        '74:DA:38:F2:BD:92':u'netcard92',
        '74:DA:38:F2:BD:BF':u'netcardbf',
        '74:DA:38:F2:BD:BD':u'netcardbd',
        '74:DA:38:F2:BD:9F':u'netcard9f',
        '74:DA:38:F2:BD:57':u'netcard57',
        }
#21- netcard
mac_label={'04:CF:8C:F4:EB:C7':1,
        '04:CF:8C:A0:FD:23':2,
        '90:97:D5:32:F9:8A':3,
        '04:CF:8C:AD:33:95': 4,
        '0C:9D:92:4F:3F:58':5,
        '7C:49:EB:18:F3:7C':6,
        '40:31:3C:BB:C6:94':7,
        '50:EC:50:02:25:A3':8,
        '44:23:7C:D5:24:B4':9,
        '3C:22:FB:80:39:FE':10,
        '28:6D:CD:01:FB:97':11,
        '6C:21:A2:C7:87:C1':12,
        '10:D0:7A:39:1B:1A':13,
        '30:6A:85:C9:F0:3B':14,
        '5C:1D:D9:5E:FF:A1':15,
        'E4:95:6E:45:10:88':16,
        '04:CF:8C:02:49:22':17,
        '04:CF:8C:B6:F6:6A':18,
        '04:CF:8C:B6:F6:84':19,
        '04:CF:8C:B4:C9:5B':20,
        '74:DA:38:F2:BD:92':21,
        '74:DA:38:F2:BD:BF':22,
        '74:DA:38:F2:BD:BD':23,
        '74:DA:38:F2:BD:9F':24,
        '74:DA:38:F2:BD:57':25,
        }



class phy_frame:
    def __init__(self,mac_addr,ftype,subtype,nomfreq,modules_index,feature):
        """
        mac_addr:str
        ftype:str
        subtype:str
        nomfreq:float
        modules_index:dict
        feature:dict
        """
        self.mac_addr=mac_addr
        self.ftype=ftype
        self.subtype=subtype
        self.nomfreq=nomfreq
        self.modules_index=modules_index
        self.feature=feature

    def __str__(self):
        return "mac_addr:{},ftype:{},subtype:{},nomfreq:{},modules_index:{},feature:{}".format(self.mac_addr,self.ftype,self.subtype,self.nomfreq,self.modules_index,self.feature)


feature_order=["freqofs","encoding","evm","iq_offset","zcr",'zcr_var','energy_var','energy_entropy_long','energy_entropy_short','energy_entropy_var','spectral_centroid','spectral_spread','spectral_entropy_long','spectral_entropy_short','spectral_entropy_var','spectral_flux_long','spectral_flux_var','spectral_centroid_var','spectral_spread_var',"snr"]
