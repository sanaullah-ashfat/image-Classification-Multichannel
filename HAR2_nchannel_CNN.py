from PIL import Image
from numpy import size, array 
from sklearn.utils import shuffle
from time import time
from sklearn.metrics import classification_report,confusion_matrix  
import os
import keras
from keras import backend as K
from sklearn.preprocessing import LabelEncoder 
from keras.engine import InputSpec
from keras.layers import Wrapper,  add,multiply ,TimeDistributed
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
import numpy as np
from glob import glob
from PIL import ImageFile
import random
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing import image
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,Activation
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.metrics import precision_recall_curve
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import label_binarize
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D,GlobalMaxPool2D
from keras.layers import Lambda, concatenate
from keras.initializers import he_normal
from keras import optimizers
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D,GlobalMaxPool2D
from keras.layers import Lambda, concatenate
from keras.initializers import he_normal
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.color import rgba2rgb
from matplotlib.animation import ArtistAnimation
from keras.models import Model
import xgboost as xgb 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def mean_pred(y_true,y_pred):
    return K.mean(y_pred)
    
def precision(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score
    
def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)

def matthews_correlation(y_true, y_pred):
    """Matthews correlation metric.
    It is only computed as a batch-wise average, not globally.
    Computes the Matthews correlation coefficient measure for quality
    of binary classification problems.
    """
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())
def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true))


def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.mean(K.sum(y_true * K.log(y_true / y_pred), axis=-1))

data1 = pd.read_csv('G:/HAR/HAR_Train_FFT_900.csv' ,header=None, sep=',')
data2 = pd.read_csv('G:/HAR/HAR_Test_FFT_900.csv' ,header=None, sep=',')
data3 = pd.read_csv('G:/HAR/HAR_Train_PWL_900.csv' ,header=None, sep=',')
data4 = pd.read_csv('G:/HAR/HAR_Test_PWL_900.csv' ,header=None, sep=',')

dff1=data1.values
dff2=data2.values
dff3=data3.values
dff4=data4.values


num_classes =6
img_width,img_height = 30,30
img_rows, img_cols = 30,30
num_epoch=2
batch_size=32

X_train_g = dff1[:,0:900]
X_train_r = dff3[:,0:900]

X_test_g = dff2[:,0:900]
X_test_r = dff4[:,0:900]

y_train= dff1[:,900]
y_test= dff2[:,900]

X_train_g=X_train_g.reshape(X_train_g.shape[0],img_rows,img_cols,1)
X_train_r=X_train_r.reshape(X_train_r.shape[0],img_rows, img_cols,1)

X_test_g=X_test_g.reshape(X_test_g.shape[0],img_rows,img_cols,1)
X_test_r=X_test_r.reshape(X_test_r.shape[0],img_rows, img_cols,1)

Y_train=np_utils.to_categorical(y_train, num_classes)
Y_test=np_utils.to_categorical(y_test,num_classes)
   
input_1 = Input(shape=(img_width,img_height,1))
input_2 = Input(shape=(img_width,img_height,1))


yy_test=y_test
FF=16
output_1 = Conv2D(FF,(2,2), activation='relu')(input_1)
output_1 = Conv2D(FF,(2,2), activation='relu')(output_1)
output_1 = BatchNormalization()(output_1)
output_1 = MaxPooling2D(pool_size=(2,2))(output_1)
output_1 = Dropout(0.25)(output_1)
output_1 = Flatten(name = 'flatten_1')(output_1)

output_2 = Conv2D(FF,(2,2), activation='relu')(input_2)
output_2 = Conv2D(FF,(2,2), activation='relu')(output_2)
output_2 = BatchNormalization()(output_2)
output_2 = MaxPooling2D(pool_size=(2,2))(output_2)
output_2 = Dropout(0.25)(output_2)
output_2 = Flatten(name = 'flatten_2')(output_2)


inputs =[input_1,input_2]
outputs =[output_1,output_2]
combine = concatenate(outputs)


output = Dense(16,activation='relu',name = 'flatten_98')(combine)
output = Dense(num_classes,activation='softmax',name = 'flatten_99')(output)

model = Model(inputs,[output])
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                        metrics=['accuracy',mean_pred,recall,precision, fmeasure, matthews_correlation,kullback_leibler_divergence,binary_crossentropy])
#'rmsprop'

model.summary()

XT=[X_test_g,X_test_r]

class prediction_history(keras.callbacks.Callback):
    def __init__(self):
        self.predhis = []
    def on_epoch_end(self, epoch, logs={}):
        self.predhis.append(model.predict(XT))

predictions=prediction_history()

#Executing the model.fit of the neural network
hist=model.fit([X_train_g,X_train_r],Y_train,epochs=num_epoch, batch_size=batch_size,verbose=1,validation_data=([X_test_g,X_test_r],Y_test),callbacks=[predictions]) 

save_test_pred =predictions.predhis

Save_pred = np.save('epoch_result',save_test_pred)

load_epoch_result = np.load('epoch_result.npy')



#hist=model.fit([X_train_g,X_train_r],Y_train,epochs=num_epoch, batch_size=batch_size,verbose=1,validation_data=([X_test_g,X_test_r],Y_test))  
score_PLSTM = model.evaluate([X_test_g,X_test_r],Y_test,verbose=0)                   
score=score_PLSTM
history=hist

print('Test score:', score[0])
print('Test accuracy:', score[1])
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
xc=range(num_epoch)
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
train_recall=hist.history['recall']
val_recall=hist.history['val_recall']
train_precision=hist.history['precision']
val_precision=hist.history['val_precision']
train_fmeasure=hist.history['fmeasure']
val_fmeasure=hist.history['val_fmeasure']
train_matthews_correlation=hist.history['matthews_correlation']
val_matthews_correlation=hist.history['val_matthews_correlation']
train_kullback_leibler_divergence=hist.history['kullback_leibler_divergence']
val_kullback_leibler_divergence=hist.history['val_kullback_leibler_divergence']
train_binary_crossentropy=hist.history['binary_crossentropy']
val_binary_crossentropy=hist.history['val_binary_crossentropy']


extract =Model(model.input,[model.get_layer("flatten_99").output,model.output])

#extract.summary()
####### test feature ########
test_feature= extract.predict([X_test_g])
np.save('DU_feature_output/test_feature',test_feature[0])
feature_test = np.load("DU_feature_output/test_feature.npy")

train_feature= extract.predict([X_train_g])
np.save('DU_feature_output/train_feature',train_feature[0])
feature_train = np.load("DU_feature_output/train_feature.npy")

lr_r=RandomForestClassifier(n_estimators=100)
lr_r.fit(feature_train,  Y_train)
accuracy_r=lr_r.score(feature_test,Y_test)
print('RandomForest',accuracy_r)

lr_x=xgb.XGBClassifier()
lr_x.fit(feature_train,  y_train)
accuracy_x=lr_x.score(feature_test,yy_test)
print('XgBoost',accuracy_x)

#Y_pred=model.predict([X_test_g,X_test_r])
#Y_pred=np.argmax(Y_pred,axis=1)
#
#yy_test = np.int64(yy_test)
#cnf_matrix = confusion_matrix(yy_test, Y_pred)
#print(cnf_matrix)
#
#






Y_pred=model.predict(XT)
Y_pred=np.argmax(Y_pred,axis=1)


cnf=(confusion_matrix(y_test,Y_pred))
print(confusion_matrix(y_test,Y_pred))



#pipe_lr_r=make_pipeline(StandardScaler(), hist)
y_score = model.predict([X_test_g,X_test_r])
y_test = label_binarize(yy_test, classes=[0, 1, 2, 3, 4, 5])
n_classes=6

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve
label1= ['Wlk','WUp','WDn','Sit','Stn','Lay']
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
#label='micro-average ROC curve (area = {0:0.2f})'
''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=label1[i].format(i, roc_auc[i]))

font={'family' : 'Tinos',
      'weight' : 'regular',
      'size':14}
plt.rc ('font', **font)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(b=None, which='both', axis='both')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (multi-class)')
plt.legend(loc="lower right");



def capcurve(y_values, y_preds_proba):
    num_pos_obs = np.sum(y_values)
    num_count = len(y_values)
    rate_pos_obs = float(num_pos_obs) / float(num_count)
    ideal = pd.DataFrame({'x':[0,rate_pos_obs,1],'y':[0,1,1]})
    xx = np.arange(num_count) / float(num_count - 1)
    
    y_cap = np.c_[y_values,y_preds_proba]
    y_cap_df_s = pd.DataFrame(data=y_cap)
    y_cap_df_s = y_cap_df_s.sort_values([1], ascending=False).reset_index(level = y_cap_df_s.index.names, drop=True)
    
    print(y_cap_df_s.head(20))
    
    yy = np.cumsum(y_cap_df_s[0]) / float(num_pos_obs)
    yy = np.append([0], yy[0:num_count-1]) #add the first curve point (0,0) : for xx=0 we have yy=0
    
    percent = 0.5
    row_index = int(np.trunc(num_count * percent))
    
    val_y1 = yy[row_index]
    val_y2 = yy[row_index+1]
    if val_y1 == val_y2:
        val = val_y1*1.0
    else:
        val_x1 = xx[row_index]
        val_x2 = xx[row_index+1]
        val = val_y1 + ((val_x2 - percent)/(val_x2 - val_x1))*(val_y2 - val_y1)
    
 #   sigma_ideal = 1 * xx[num_pos_obs - 1 ] / 2 + (xx[num_count - 1] - xx[num_pos_obs]) * 1
  #  sigma_model = integrate.simps(yy,xx)
  #  sigma_random = integrate.simps(xx,xx)
    
  #  ar_value = (sigma_model - sigma_random) / (sigma_ideal - sigma_random)
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    ax.plot(ideal['x'],ideal['y'], color='grey', label='Perfect Model')
    ax.plot(xx,yy, color='red', label='User Model')
    ax.plot(xx,xx, color='blue', label='Random Model')
    ax.plot([percent, percent], [0.0, val], color='green', linestyle='--', linewidth=1)
    ax.plot([0, percent], [val, val], color='green', linestyle='--', linewidth=1, label=str((round(val*100,2)))+'% .of positive obs at '+str(percent*100)+'%')
    
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.25)
   # plt.title("CAP Curve - a_r value ="+str(ar_value))
    plt.xlabel('% of the data')
    plt.ylabel('% of positive obs')
    plt.legend()
    #plt.savefig(directory +'cap_graph.png')
    plt.show()
    
    
capcurve(y_test,y_score)    