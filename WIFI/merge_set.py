import os
import optparse
import re


mac_base={'04:CF:8C:F4:EB:C7':0,
        '04:CF:8C:A0:FD:23':0,
        '90:97:D5:32:F9:8A':0,
        '04:CF:8C:AD:33:95': 0,
        '0C:9D:92:4F:3F:58':0,
        '7C:49:EB:18:F3:7C':0,
        '40:31:3C:BB:C6:94':0,
        '50:EC:50:02:25:A3':0,
        '44:23:7C:D5:24:B4':0,
        '3C:22:FB:80:39:FE':0,
        '28:6D:CD:01:FB:97':0,
        '6C:21:A2:C7:87:C1':0,
        '10:D0:7A:39:1B:1A':0,
        '30:6A:85:C9:F0:3B':0,
        '5C:1D:D9:5E:FF:A1':0,
        'E4:95:6E:45:10:88':0,
        '04:CF:8C:02:49:22':0,
        }

def mv_flie(s_folder,o_path,start,num):
    prev_base=os.getcwd()
    os.chdir(s_folder)
    base=os.getcwd()
    #print(base)
    mac_name=base.split('/')[-1]
    #print(mac_name)
    o_path=os.path.join(o_path,mac_name)
    if not os.path.exists(o_path):
    # 如果不存在则创建目录
    #创建目录操作函数
        os.makedirs(o_path)
    for i in findAllFile(s_folder):
        if mac_base[mac_name]==num:
            break
        mac_base[mac_name]+=1
        # mv i d_path/i+start
        #print(i)
        file_name=i.split('/')[-1]
        #print(file_name)
        os.system('mv '+i+' '+os.path.join(o_path,str(mac_base[mac_name])+'.npy'))
        #print(vector)



    os.chdir(prev_base)
    return


def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if re.match(r'.*\d.*', f):
                fullname = os.path.join(root, f)
                yield fullname



def main():
    parser=optparse.OptionParser("'usage -s <filefloder1,filefloder2>,  -o <output folder >")
    parser.add_option('-s',dest='s_path',type='string',help='specify source path')
    parser.add_option('-o',dest='o_path',type='string',help='specify output path')
    parser.add_option('-n',dest='num_sample',type='int',help='specify num_samples to move')
    (options,args)=parser.parse_args()
    if (options.s_path==None):
        print(parser.usage)
        exit(0)
    s_paths=[os.path.join(os.getcwd(),i) for i in options.s_path.split(',')]
    o_path=os.path.join(os.getcwd(),options.o_path)
    num_sample=options.num_sample

   

    for s_path in s_paths:
        for dirname in os.listdir(s_path):
            if (dirname in mac_base) and mac_base[dirname] <num_sample:
                s_folder=os.path.join(s_path,dirname)
                mv_flie(s_folder,o_path,mac_base[dirname],num_sample)




if __name__ == '__main__':
    main()
