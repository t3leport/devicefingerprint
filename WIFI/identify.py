#导入MLP神经网络
from sklearn.neural_network import MLPClassifier
#导入红酒数据集
from sklearn.datasets import load_wine
#导入数据集拆分工具
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import os,re
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import optparse
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
import time
from utility import *



def identify(pathname,modelname,c_features):
    X,Y=load_samples(pathname,c_features)
    robustscaler=preprocessing.RobustScaler()      #建立MaxAbsScaler对象
    X=robustscaler.fit_transform(X)  #MaxAbScaler标准化处理
    model = joblib.load(modelname)
    y_pred=model.predict(X)
    print("*"*10+modelname+"*"*10+pathname)
    print(classification_report(Y, y_pred, labels=None, target_names=None, sample_weight=None, digits=2))
    print(confusion_matrix(Y, y_pred, labels=None, sample_weight=None))


if __name__ == '__main__':
    parser=optparse.OptionParser("'usage -p <path to folder>, -l <save model >,-c <choose features> ")
    parser.add_option('-p',dest='path',type='string',help='specify the path ')
    parser.add_option('-l',dest='model',help='the name of  loading model ')
    parser.add_option('-c',dest='c_features',help='choose features ')
    (options,args)=parser.parse_args()
    if (options.path==None or options.model==None):
        print(parser.usage)
        exit(0)
    pathname=options.path
    modelname=options.model
    c_features=options.c_features
    identify(pathname,modelname,c_features)
    
