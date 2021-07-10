from ble_uti import *
import optparse
file_name='ioszyk209.pcap'
samples_count=50
import time
save_time=time.strftime("%m-%d-%H:%M", time.localtime())

'''
ble_labels=[('SmartBracelets','xiaomi'),('SmartBracelets','huawei'),('SmartBracelets','garmin'),('SmartBracelets','ADZ'),('SmartBracelets','polar'),
            ('smartwatch','garmin'),('smartwatch','xiaomi'),('smartwatch','polar'),
            ('SmartPulseOximeter','konsung'),('SmartPulseOximeter','viatom'),
            ('laptop','apple'),
            ]

'''
mac_labels_dict={
            ('SmartBracelets','xiaomi'):[b'f9733d65c8ef'],
            ('SmartBracelets','huawei'):[b'0c839ac4c549',
            b'5c78f85b2860',
            b'0c839ac4be56',
            b'0c839ac4c549',
            b'0c839ac4bd17',
            b'0c839ac4c506'],
            ('SmartBracelets','garmin'):[b'fb90fc4dd08b'],
            ('SmartBracelets','ADZ'):[b'cacbf45abd5c'],
            ('SmartBracelets','polar'):[b'a09e1a72fff4'],
            ('smartwatch','garmin'):[b'cb0c6dae8615'],
            ('smartwatch','xiaomi'):[b'5ce50cf7bcaa'],
            ('smartwatch','polar'):[b'a09e1a66ad16'],
            ('SmartPulseOximeter','konsung'):[b'628300000333'],
            ('SmartPulseOximeter','viatom'):[b'c793be04ebc0'],
            ('laptop','apple'):[b'6dcaa5f15ba7',b'605084f8b186',b'406ce6c987eb'],
            ('mobilephone','apple'):[
                b'6db36ca76073',b'5812d79c4079',
                b'49b0b72208ae'
            ],
            ('headphone','sony'):[b'38184c2407bc'],
            ('SmartHumidifier','xiaomi'):[b'04cf8cad3396'],
            ('desktopComputer','microsoft'):[b'61bcdd2ae23e',
            b'2ebaacea1f24']

}

if __name__ == "__main__":
    parser=optparse.OptionParser("'usage -f <pcap file path> ")
    parser.add_option('-f',dest='filename',type='string',help='specific pcap file')
    (options,args)=parser.parse_args()
    if (options.filename==None):
        print(parser.usage)
        exit(0)
    file_name=options.filename
    init_mac_labels(mac_labels_dict)
   
    all_pcap,mac_count=collect_all_pcap(file_name)
    mac_list=[key for key in mac_count]
    print(len(all_pcap))

    #form_vector(all_pcap,mac_list,bin_size)
    #5个bin_size只有4个adv_interval
    #bins=form_sample_bin(all_pcap,mac_count,bin_size,mac_list,tor)
    print(mac_count)
    #print(len(bins))
    form_vector("ble_train"+file_name.split('.')[0],all_pcap,1,200000)
    
    '''
    for pcap in all_pcap:
        print(pcap)
    '''
