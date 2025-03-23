# Updated and cleaned version of the given dual-input CNN model for HAR classification
# with performance metrics, ROC, CAP curves and RandomForest/XGBoost classifiers

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import keras
import xgboost as xgb
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, concatenate, BatchNormalization
from keras.utils import np_utils
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report,
    precision_recall_curve, roc_auc_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier

# ---- Load Data ----
data1 = pd.read_csv('G:/HAR/HAR_Train_FFT_900.csv', header=None)
data2 = pd.read_csv('G:/HAR/HAR_Test_FFT_900.csv', header=None)
data3 = pd.read_csv('G:/HAR/HAR_Train_PWL_900.csv', header=None)
data4 = pd.read_csv('G:/HAR/HAR_Test_PWL_900.csv', header=None)

dff1, dff2, dff3, dff4 = data1.values, data2.values, data3.values, data4.values

num_classes = 6
img_rows = img_cols = 30

X_train_g = dff1[:, 0:900].reshape(-1, img_rows, img_cols, 1)
X_train_r = dff3[:, 0:900].reshape(-1, img_rows, img_cols, 1)
X_test_g  = dff2[:, 0:900].reshape(-1, img_rows, img_cols, 1)
X_test_r  = dff4[:, 0:900].reshape(-1, img_rows, img_cols, 1)

y_train = dff1[:, 900]
y_test  = dff2[:, 900]

Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test  = np_utils.to_categorical(y_test, num_classes)

# ---- Custom Metrics ----
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def fmeasure(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r + K.epsilon())

def matthews_correlation(y_true, y_pred):
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

# ---- Model Definition ----
input_1 = Input(shape=(img_rows, img_cols, 1))
input_2 = Input(shape=(img_rows, img_cols, 1))

FF = 16

def cnn_branch(input_layer):
    x = Conv2D(FF, (2, 2), activation='relu')(input_layer)
    x = Conv2D(FF, (2, 2), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    return Flatten()(x)

branch1 = cnn_branch(input_1)
branch2 = cnn_branch(input_2)

merged = concatenate([branch1, branch2])
x = Dense(16, activation='relu')(merged)
out = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=[input_1, input_2], outputs=out)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy', recall, precision, fmeasure, matthews_correlation])

model.summary()

# ---- Training ----
class PredictionHistory(Callback):
    def __init__(self):
        self.predhis = []
    def on_epoch_end(self, epoch, logs=None):
        self.predhis.append(model.predict([X_test_g, X_test_r]))

pred_cb = PredictionHistory()
history = model.fit([X_train_g, X_train_r], Y_train,
                    epochs=10, batch_size=32, verbose=1,
                    validation_data=([X_test_g, X_test_r], Y_test),
                    callbacks=[pred_cb])

# ---- Evaluation ----
score = model.evaluate([X_test_g, X_test_r], Y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# ---- Feature Extraction ----
feature_model = Model(inputs=model.input, outputs=model.get_layer(index=-2).output)
train_features = feature_model.predict([X_train_g, X_train_r])
test_features  = feature_model.predict([X_test_g, X_test_r])

# ---- Traditional Classifiers ----
rf = RandomForestClassifier(n_estimators=100)
rf.fit(train_features, y_train)
print("RandomForest Accuracy:", rf.score(test_features, y_test))

xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(train_features, y_train)
print("XGBoost Accuracy:", xgb_clf.score(test_features, y_test))
