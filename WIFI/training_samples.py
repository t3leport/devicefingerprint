#导入MLP神经网络
from sklearn.neural_network import MLPClassifier
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
from utility import *
from sklearn.ensemble import RandomForestClassifier
import math
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate

scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_macro': 'recall_macro',
           'f1_macro':'f1_macro',
           'f1_micro':'f1_micro'}



def cross_val(X,Y,model_name,model):
    scores = cross_validate(model,X,Y,cv=10,scoring=scoring)
    print(model_name+" 10 10-folder")
    #print(scores)
    for key in scoring:
        print("average "+key)
        for j in scores:
            if j=='test_'+key:
                print(scores[j].mean())


def print_confusion_matrix(X,Y,model_name,model,save):
    X_train,X_test,y_train,y_test = train_test_split(X,Y,random_state=0,test_size=0.3)
    pred=model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print("*"*10+model_name+"*"*10)
    print(classification_report(y_test, y_pred, labels=None, target_names=None, sample_weight=None, digits=4))
    conf_mat=confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
    #print(conf_mat)
    #plot_confusion_matrix(conf_mat, normalize=False,classes=[str(i+1) for i in range(len(set(Y)))],title='Confusion Matrix')
    if save==1:
        joblib.dump(pred, model_name+'_'+str(len(X_train[0]))+'_'+pathname.split('/')[-2]+'.model')
        print('saved as '+model_name+'_'+str(len(X_train[0]))+'_'+pathname.split('/')[-2]+'.model')


def train_mlp(pathname,c_features,save=0):
    #下面我们拆分数据集
    X,Y=load_samples(pathname,c_features)
    robustscaler=preprocessing.RobustScaler()      #建立MaxAbsScaler对象
    X=robustscaler.fit_transform(X)  #MaxAbScaler标准化处理
    #接下来定义分类器 设置神经网络2个节点数为10的隐藏层 hidden_layer_sizes=[10,10]
    mlp_clf__tuned_parameters = {"hidden_layer_sizes": [(100,), (100, 30),(128,100,50),(128,128,128)],
                             "solver": ['adam', 'sgd', 'lbfgs'],
                             "max_iter": [20,200,1000] }
    #grid=GridSearchCV(MLPClassifier(), mlp_clf__tuned_parameters, n_jobs=6)
    #grid.fit(X, Y)
    #print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

    mlp = MLPClassifier(solver='adam',max_iter=3000,hidden_layer_sizes=[128,128,128],activation='relu')
    #10-folder
    cross_val(X,Y,"MLP",mlp)
    print_confusion_matrix(X,Y,"MLP",mlp,save)
   

def train_rf(pathname,c_features,save=0):
    X,Y=load_samples(pathname,c_features)
    robustscaler=preprocessing.RobustScaler()      #建立MaxAbsScaler对象
    X=robustscaler.fit_transform(X)  #MaxAbScaler标准化处理
    print("feature number "+str(len(X[0])))
    grid = GridSearchCV(RandomForestClassifier(), param_grid={"n_estimators":range(10,121,10)}, cv=10)
    grid.fit(X, Y)
    print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
    #clf2 = RandomForestClassifier(n_estimators=10,max_features='sqrt', max_depth=None,min_samples_split=2, bootstrap=True)
    clf2 = RandomForestClassifier(n_estimators=100, max_depth=None, bootstrap=True)
    cross_val(X,Y,"random_forest",clf2)
    print_confusion_matrix(X,Y,"random_forest",clf2,save)


def train_nb(pathname,c_features,save=0):
    X,Y=load_samples(pathname,c_features)
    robustscaler=preprocessing.RobustScaler()      #建立MaxAbsScaler对象
    X=robustscaler.fit_transform(X)  #MaxAbScaler标准化处理
    print("feature number "+str(len(X[0])))
       
    gnb=GaussianNB()

    cross_val(X,Y,"navie_bayes",gnb)

    print_confusion_matrix(X,Y,"navie_bayes",gnb,save)
    

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

def train_svm(pathname,c_features,save=0):
    X,Y=load_samples(pathname,c_features)
    robustscaler=preprocessing.RobustScaler()      #建立MaxAbsScaler对象
    X=robustscaler.fit_transform(X)  #MaxAbScaler标准化处理
    grid = GridSearchCV(SVC(), param_grid={"C":[0.1, 1, 10], "gamma": [1, 0.1, 0.01]}, cv=10)
    grid.fit(X, Y)
    print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

    sv=SVC(gamma=0.01,C=10)
    cross_val(X,Y,"SVM",sv)
    print_confusion_matrix(X,Y,"SVM",sv,save)

from sklearn.neighbors import KNeighborsClassifier

def train_knn(pathname,c_features,save=0):
    X,Y=load_samples(pathname,c_features)
    robustscaler=preprocessing.RobustScaler()      #建立MaxAbsScaler对象
    X=robustscaler.fit_transform(X)  #MaxAbScaler标准化处理
    grid = GridSearchCV(KNeighborsClassifier(), param_grid={"n_neighbors":[1,3,5,7,9,11,13,15,17,19,21,23,25]}, cv=10)
    grid.fit(X, Y)
    print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
    knn = KNeighborsClassifier(n_neighbors=1)
    cross_val(X,Y,"KNN",knn)
    print_confusion_matrix(X,Y,"KNN",knn,save)



from sklearn.tree import DecisionTreeClassifier

def train_decisiontree(pathname,c_features,save=0):
    X,Y=load_samples(pathname,c_features)
    robustscaler=preprocessing.RobustScaler()      #建立MaxAbsScaler对象
    X=robustscaler.fit_transform(X)  #MaxAbScaler标准化处理
    '''
    grid = GridSearchCV(KNeighborsClassifier(), param_grid={"n_neighbors":[1,3,5,7,9,11,13,15,17,19,21,23,25]}, cv=10)
    grid.fit(X, Y)
    print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
    '''
    dt = DecisionTreeClassifier()
    cross_val(X,Y,"DecisionTree",dt)
    print_confusion_matrix(X,Y,"DecisionTree",dt,save)

from sklearn.ensemble import GradientBoostingClassifier
def train_gbdt(pathname,c_features,save=0):
    X,Y=load_samples(pathname,c_features)
    robustscaler=preprocessing.RobustScaler()      #建立MaxAbsScaler对象
    X=robustscaler.fit_transform(X)  #MaxAbScaler标准化处理
    '''
    
    grid = GridSearchCV(GradientBoostingClassifier(), param_grid={"n_estimators":range(10,121,10)}, cv=10)
    grid.fit(X, Y)
    print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
    '''
    gbdt = GradientBoostingClassifier(n_estimators=100)
    cross_val(X,Y,"gbdt",gbdt)
    print_confusion_matrix(X,Y,"gbdt",gbdt,save)


from sklearn.linear_model import LogisticRegressionCV
def train_lr(pathname,c_features,save=0):
    X,Y=load_samples(pathname,c_features)
    robustscaler=preprocessing.RobustScaler()      #建立MaxAbsScaler对象
    X=robustscaler.fit_transform(X)  #MaxAbScaler标准化处理
  
    '''
    grid = GridSearchCV(LogisticRegressionCV(), param_grid={"solver":["liblinear","lbfgs","newton-cg","sag"],"tol":[0.001,0.01,0.1]}, cv=10)
    grid.fit(X, Y)
    print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
    '''
    lr = LogisticRegressionCV(penalty="l2",solver="lbfgs",max_iter=100000)

    cross_val(X,Y,"LogisticRegression",lr)
    print_confusion_matrix(X,Y,"LogisticRegression",lr,save)


#03-12 计算特征均值和方差

def cal_feature_avg_var(phyframes,mac_iot):
    for mac in mac_iot:
        x_freqofs=[]
        nomfreq=0
        for phyframe in phyframes:
            if phyframe.mac_addr==mac:
                x_freqofs.append(phyframe.feature['freqofs'])
        print("mac: {}, freqofs_avg: {} ,var:{}".format(mac,sum(x_freqofs)/len(x_freqofs)*2.432e9*1e6,np.var(x_freqofs)))

import pandas as pd
from minepy import MINE
def print_mic(X,Y,c_features):
    feature_mic=dict()
    df=pd.DataFrame(X,columns=c_features.split(','))
    m=MINE()
    #print(df['zcr'])
    for key in c_features.split(','):
        m.compute_score(df[key],Y)
        #print(key+':'+str(m.mic()))
        feature_mic[key]=m.mic()
    dic=sorted(feature_mic.items(),key=lambda d:d[1],reverse=True)
    index=1
    for i in dic:
        print(str(index)+" "+i[0]+":"+str(i[1]))
        index+=1
    

if __name__ == '__main__':
    parser=optparse.OptionParser("'usage -p <path to folder>, -s <save model > ")
    parser.add_option('-p',dest='path',type='string',help='specify the path ')
    parser.add_option('-c',dest='features',type='string',help='choose features to training')
    parser.add_option('-s',dest='save',help='save model or not ',action='store')
    (options,args)=parser.parse_args()
    if (options.path==None):
        print(parser.usage)
        exit(0)
    pathname=options.path
    c_features=options.features
    save=False if options.save==None else True

    
    #feature selection
    X,Y=load_samples(pathname,c_features)
    print_mic(X,Y,c_features)
    
    train_nb(pathname,c_features,save)
    
    train_rf(pathname,c_features,save)
    
    train_svm(pathname,c_features,save)
    train_decisiontree(pathname,c_features,save)
    train_knn(pathname,c_features,save)
    
    train_gbdt(pathname,c_features,save)
    
    train_lr(pathname,c_features,save)
    train_mlp(pathname,c_features,save)
    

    
    mac_iot=['04:CF:8C:A0:FD:23','90:97:D5:32:F9:8A','04:CF:8C:AD:33:95',
        '0C:9D:92:4F:3F:58','7C:49:EB:18:F3:7C','40:31:3C:BB:C6:94','50:EC:50:02:25:A3',
        '44:23:7C:D5:24:B4','04:CF:8C:02:49:22','28:6D:CD:01:FB:97','6C:21:A2:C7:87:C1',
        '10:D0:7A:39:1B:1A','30:6A:85:C9:F0:3B','5C:1D:D9:5E:FF:A1','3C:22:FB:80:39:FE']
    '''
    mac_iot=['74:DA:38:F2:BD:92','74:DA:38:F2:BD:BF','74:DA:38:F2:BD:BD','74:DA:38:F2:BD:9F','74:DA:38:F2:BD:57',
        '08:BE:AC:0F:05:DB','74:DA:38:F2:BD:C7','08:BE:AC:0F:05:DD','08:BE:AC:0F:06:25','08:BE:AC:0F:05:D5']
    '''
    phy_frames=[]
    phy_frames=load_phyframe(pathname,mac_iot,c_features)
    
    #plot_feature=['zcr','spectral_spread_short','energy_entropy_long','spectral_centroid_long','spectral_spread_long','spectral_entropy_long','energy_entropy','spectral_centroid,spectral_spread','spectral_entropy','energy_entropy_var']
    feature_boxplot(phy_frames,['sync_correlation','phaseerror','magerror'],n=200)
    cal_feature_avg_var(phy_frames,mac_iot)
