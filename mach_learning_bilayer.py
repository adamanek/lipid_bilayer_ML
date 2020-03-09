#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:04:46 2019

@author: adam
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras.utils import np_utils
import config
from clr_callback import CyclicLR
import seaborn as sns

def one_hot_encode_object_array(arr):
    '''One hot encode a numpy array of objects (e.g. strings)'''
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))

np.set_printoptions(suppress=True)
DOPC_train = pd.read_csv('output/training_set_DOPC.csv', header = None)
DPPC_train = pd.read_csv('output/training_set_DPPC.csv', header = None)
dataset_train = pd.concat([DOPC_train,DPPC_train], axis = 0).values
X = dataset_train[:,0:3]
y = dataset_train[:,3]

#change classes from strings to binary matrix
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
y = np_utils.to_categorical(y)

#split the simulated data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle='True') 

opt = SGD(lr=config.MIN_LR, momentum=0.9)

#making the ML model
model = Sequential()
model.add(Dense(12,input_dim=3, activation='relu'))
#model.add(Dropout(0.05))
model.add(Dense(12,activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', 
              optimizer = 'adam', 
              metrics=['accuracy', 'binary_crossentropy'])


# initialize the cyclical learning rate callback
print("[INFO] using '{}' method".format(config.CLR_METHOD))
clr = CyclicLR(
	mode=config.CLR_METHOD,
	base_lr=config.MIN_LR,
	max_lr=config.MAX_LR,
	step_size= config.STEP_SIZE * (X_train.shape[0] // config.BATCH_SIZE))

#fitting the model
baseline_history = model.fit(X_train, 
                             y_train, 
                             epochs=config.NUM_EPOCHS,
                             callbacks=[clr], 
                             batch_size=config.BATCH_SIZE,
                             #steps_per_epoch=X_train.shape[0] // config.BATCH_SIZE,
                             validation_data=(X_test, y_test),
                             verbose=1
                             )
model.summary()
pred_train = model.predict(X_train)

scores = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   
 
pred_test= model.predict(X_test, batch_size=config.BATCH_SIZE)
scores2 = model.evaluate(X_test, y_test, verbose=0)

print(classification_report(y_test.argmax(axis=1),
	pred_test.argmax(axis=1), target_names=config.CLASSES))

print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))    

#Using real Boris Bike data to see how accurate the model is
DOPC_test = pd.read_csv('output/test_set_DOPC.csv', header = None)
DPPC_test = pd.read_csv('output/test_set_DPPC.csv', header = None)
dataset_test = pd.concat([DOPC_test,DPPC_test], axis = 0)
b_r = dataset_test.values[:,0:3]
pred_bike = model.predict(b_r)
predictions = model.predict_classes(b_r)
prediction_ = np.argmax(to_categorical(predictions), axis = 1)
prediction_ = encoder.inverse_transform(prediction_)
unique_elements, count_elements = np.unique(prediction_, return_counts=True)

#Plotting prediction on real set
DOPC_pos= pd.read_csv('output/positions_DOPC.csv', header = None, names = ['X','Y','Lipid type'])
DPPC_pos = pd.read_csv('output/positions_DPPC.csv', header = None, names = ['X','Y','Lipid type'])
dataset_pos = pd.concat([DOPC_pos,DPPC_pos], axis = 0).values
pred_df = pd.DataFrame(predictions, index = None).values

dataset_whole = pd.DataFrame(np.concatenate([dataset_pos,pred_df], axis = 1), columns=['X','Y','Lipid Type','Order'])
sns_plot = sns.relplot(x='X',y='Y',hue='Order', data = dataset_whole, s =10, kind = 'scatter')

sns_plot.savefig('output/Order_test.png',dpi=300)

def plot_history(histories, key='binary_crossentropy'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val Loss')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train Loss')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.plot(baseline_history.history['acc'], label='train acc')
  plt.plot(baseline_history.history['val_acc'], label='test acc')
  plt.legend()

  plt.xlim([0,max(history.epoch)])


plot_history([('baseline', baseline_history)])



# construct a plot that plots and saves the training history
# =============================================================================
# N = np.arange(0, config.NUM_EPOCHS)
# plt.style.use("ggplot")
# plt.figure(figsize=(16,10))
# plt.plot(N, baseline_history.history["loss"], label="train_loss")
# plt.plot(N, baseline_history.history["val_loss"], label="val_loss")
# plt.plot(N, baseline_history.history["acc"], label="train_acc")
# plt.plot(N, baseline_history.history["val_acc"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig(config.TRAINING_PLOT_PATH)
# =============================================================================
 
# plot the learning rate history
N = np.arange(0, len(clr.history["lr"]))
plt.figure(figsize=(16,10))
plt.plot(N, clr.history["lr"])
plt.title("Cyclical Learning Rate (CLR)")
plt.xlabel("Training Iterations")
plt.ylabel("Learning Rate")
plt.savefig(config.CLR_PLOT_PATH)
