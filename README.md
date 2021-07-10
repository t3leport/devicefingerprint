# DeviceFingerprint
DeviceFingerprint for BLE or WiFi device

WiFi
采集数据
python capture_samples.py -c 2432000000 >11_20.flag

处理flag文件
1. snr 小于10的样本删掉python3 edit_flag.py -f 11_25.flag -s 10
2. 查看有哪些mac地址 python3 edit_flag.py -f 11_25.flag -p xxx
3. 删除mac地址 python3 edit_flag.py -f 11_25.flag -m mac1,mac2

生成数据集
python3 sort_samples.py	
python3 merge_set.py -s 12-04-09\:49/,12-03-17\:53,12-02-16\:11,11-25-09\:35 -o iot_dev_2 -n 100

训练
python3 training_samples.py -p 11-25-09\:35/ -c freqofs,evm,encoding,iq_offset
选取储存的前4个特征训练并存储模型
python3 training_samples.py -p 11-25-09\:35/ -c freqofs,evm,encoding,iq_offset  -s 1

识别
python3 identify.py -p 11-24-15\:02/ -l navie_bayes_11-23-19:36.model -c freqofs,encoding,evm,iq_offset

BLE
python3 ble_gather.py -f .pcap
python3 ble_train.py
python3 ble_test.py
