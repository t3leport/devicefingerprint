import numpy as np
import os
from sklearn.metrics import classification_report
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

FEATURE_LEN = 100

#TEST_FOLDER = "ble_trainble_all_15/"
TEST_FOLDER = "ble_test_all/"
test_files = os.listdir(TEST_FOLDER)
len_test = len(test_files)
model = keras.models.load_model('project_lab_model/one_packet_model_at_epoch_35.h5')

scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_macro': 'recall_macro',
           'f1_macro':'f1_macro',
           'f1_micro':'f1_micro'}


def plot_confusion_matrix(cm, classes,
                          normalize=True,
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
 
    
    plt.title(title,fontsize=20)
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
    plt.ylabel('True label',fontsize=17)
    plt.xlabel('Predicted label',fontsize=17)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.show()



Y_pre = [0] * len_test
X_tru = [0] * len_test
for i in range(len(test_files)):
    try:
        test_data = np.load(TEST_FOLDER + test_files[i])
    except:
        continue
    arr_temp = test_data[:-1]
    label = np.int64(test_data[-1])
#     label = int(test_data[-1].tolist()[0])
    arr=np.array([float(i) for i in arr_temp]).reshape([1, 2*FEATURE_LEN])
    predict = model.predict_classes(arr).tolist()[0]
    Y_pre[i] = predict
    X_tru[i] = label
    #print ('predict: {}, label: {}'.format(predict, label))
#print (classification_report(X_tru, Y_pre, digits=6))

print(classification_report(X_tru, Y_pre, labels=None, target_names=None, sample_weight=None, digits=4))

conf_mat=confusion_matrix(X_tru, Y_pre, labels=None, sample_weight=None)
plot_confusion_matrix(conf_mat, normalize=True,classes=[str(i+1) for i in range(15)],title='Confusion Matrix')



