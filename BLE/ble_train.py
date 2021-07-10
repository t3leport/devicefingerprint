from ble_uti import *

train_pathname="/Users/hubuyi/Documents/mymaster/ble_lab/ble_trainble_all_15"
#test_pathname="/home/paraodx/ble_lab/ble_train_test"
#train_pathname="/Users/hubuyi/Documents/mymaster/rf_lab/Raw_data"


def train(X,Y):
    #下面我们拆分数据集
    X_train,X_test,y_train,y_test = train_test_split(X,Y,random_state=0,test_size=0.3)
    #接下来定义分类器 设置神经网络2个节点数为10的隐藏层 hidden_layer_sizes=[10,10]
    mlp = MLPClassifier(solver='lbfgs',max_iter=1500,hidden_layer_sizes=[128,128,128],activation='relu')
    scores = cross_val_score(mlp,X,Y,cv=10,scoring='accuracy')
    print("10折交叉验证: "+str(scores.mean()))

    mlp.fit(X_train,y_train)
    y_pred = mlp.predict(X_test)
    print(accuracy_score(y_test,y_pred))
    print(classification_report(y_test, y_pred, labels=None, target_names=None, sample_weight=None, digits=2))
    print(confusion_matrix(y_test, y_pred, labels=None, sample_weight=None))

from keras.models import Sequential
from keras.layers import Dense, Activation
def train_ker(X,Y):
    model = Sequential([
    	Dense(128, input_shape=(100,)),
    	Activation('softmax'),
   	    Dense(128),
    	Activation('softmax'),],
        Dense(128,output_shape=(128,len(Y))),
        Activation('sigmoid'))

if __name__ == '__main__':

    X,Y=get_X_Y(train_pathname)
    train(X,Y)
    #train_mlp_ie_classifier(X,Y,train_pathname)
    #train_nb_with_ie_mlp_out_and_adv_intervals(train_pathname,X,Y)

    
    #identify(test_pathname,train_pathname)



