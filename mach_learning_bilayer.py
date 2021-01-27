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
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Conv2D, Flatten
from keras.utils import np_utils
import config
from clr_callback import CyclicLR
import seaborn as sns
import keras

def one_hot_encode_object_array(arr):
    '''One hot encode a numpy array of objects (e.g. strings)'''
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))

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


np.set_printoptions(suppress=True)
DOPC_train = pd.read_csv('output/train_set_DOPC_CHOL_mean.csv', header = None)
DPPC_train = pd.read_csv('output/train_set_DPPC_CHOL_mean.csv', header = None)
dataset_train = pd.concat([DOPC_train,DPPC_train], axis = 0).values
X = dataset_train[:,0:6]
y = dataset_train[:,6]


#split the simulated data into training and testing data
#D_label = np.load('output/train_set_DPPC_ord_3D_label.npy', allow_pickle = True)
#O_label = np.load('output/train_set_DOPC_ord_3D_label.npy', allow_pickle = True)
#O_CHOL_label = np.load('output/train_set_DOPC_CHOL_3D_label.npy', allow_pickle = True)
#
#
#O = np.load('output/train_set_DOPC_ord_3D_new.npy', allow_pickle = True)
#D = np.load('output/train_set_DPPC_ord_3D.npy', allow_pickle = True)
#O_CHOL = np.load('output/train_set_DOPC_CHOL_3D.npy', allow_pickle = True)
#
#y= np.concatenate((D_label, O_CHOL_label))
#X = np.concatenate((D,O_CHOL), axis = 1)

#change classes from strings to binary matrix
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
y = np_utils.to_categorical(y)



#X = np.transpose(X, (1,2,0))
#all_indices = list(range(len(X)))
#train_indices, test_indices = train_test_split(all_indices, test_size=0.30, shuffle='True')
#X_train = X[train_indices,:,:]
#X_test = X[test_indices,:,:]
#y_train = y[train_indices]
#y_test = y[test_indices]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle='True') 

#Pre-process input variables with a scaler
scaler = StandardScaler()
# fit scaler on training dataset
scaler.fit(X_train)
# transform training dataset
X_train = scaler.transform(X_train)
# transform test dataset
X_test = scaler.transform(X_test)


opt = SGD(lr=config.MIN_LR, momentum=0.9)
#making the ML model
model = Sequential()
#model.add(Conv2D(32,kernel_size=(3,3),input_shape=(X_train.shape), activation='relu'))
#model.add(Dense(24,input_shape = (199,3), activation = 'relu'))
model.add(Dense(24,input_dim = 6, activation = 'relu'))

#model.add(Dropout(0.1))

model.add(Dropout(0.2))
#model.add(Dense(24,activation='relu'))
model.add(Dense(12, activation = 'relu'))
#model.add(Dropout(0.05))
#model.add(Flatten())
#model.add(Dropout(0.05))
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
plot_history([('baseline', baseline_history)])
plt.savefig('output/Metrics_model.png',dpi=300)

model.save('output/model_mean_stdev_DPPC_DOPC.h5')
#Using real Boris Bike data to see how accurate the model is
model = keras.models.load_model('output/model_mean_stdev_DPPC_DOPC.h5')

B_leaflet0 = np.load('output/di4_protein_mean_leaflet0_30.npy', allow_pickle=True)
#B_leaflet0 = np.transpose(B_leaflet0, (1,2,0))
B_leaflet0_label = np.load('output/di4_protein_mean_leaflet0_labels_30.npy', allow_pickle=True)

#CGtest = pd.read_csv('output/CG_dian_leaflet1.csv', header = None)
#DPPC_test = pd.read_csv('output/test_set_DPPC.csv', header = None)
#dataset_test = pd.concat([DOPC_test,DPPC_test], axis = 0)
b_r = scaler.transform(B_leaflet0)
pred_bike = model.predict(b_r)
predictions = model.predict_classes(b_r)
prediction_ = np.argmax(to_categorical(predictions), axis = 1)
prediction_ = encoder.inverse_transform(prediction_)
unique_elements, count_elements = np.unique(prediction_, return_counts=True)

#Plotting prediction on real set
CG_pos= pd.read_csv('output/di4_protein_mean_positions_leaflet0_30.csv', header = None, names = ['X','Y','Lipid type','resid'])
#DPPC_pos = pd.read_csv('output/positions_DPPC.csv', header = None, names = ['X','Y','Lipid type'])
#dataset_pos = pd.concat([DOPC_pos,DPPC_pos], axis = 0).values
pred_df = pd.DataFrame(prediction_, index = None).values

dataset_whole = pd.DataFrame(np.concatenate([CG_pos.values,pred_df], axis = 1), columns=['X','Y','Lipid Type','resid','Order'])
g = sns.relplot(x='X',y='Y',hue='Order', data = dataset_whole, s =10, kind = 'scatter')
g.fig.set_size_inches(15,15)


dataset_whole.to_csv('output/di4_protein_mean_leaflet0_30_dataset.csv')
g.savefig('output/di4_protein_mean_leaflet0_30.png',dpi=300)





# construct a plot that plots and saves the training history
# =============================================================================
N = np.arange(0, config.NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure(figsize=(16,10))
plt.plot(N, baseline_history.history["loss"], label="train_loss")
plt.plot(N, baseline_history.history["val_loss"], label="val_loss")
plt.plot(N, baseline_history.history["acc"], label="train_acc")
plt.plot(N, baseline_history.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(config.TRAINING_PLOT_PATH)
# =============================================================================
 
# plot the learning rate history
N = np.arange(0, len(clr.history["lr"]))
plt.figure(figsize=(16,10))
plt.plot(N, clr.history["lr"])
plt.title("Cyclical Learning Rate (CLR)")
plt.xlabel("Training Iterations")
plt.ylabel("Learning Rate")
plt.savefig(config.CLR_PLOT_PATH)

from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patches as mpatches
# find min/max values for normalization
minima = min(predictions)
maxima = max(predictions)

# normalize chosen colormap
norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)

mapper = cm.ScalarMappable(norm=norm,cmap=cm.seismic)
mpl.rcParams.update({'font.size': 20})

patch = [mpatches.Patch(color = mapper.to_rgba(minima)), mpatches.Patch(color = mapper.to_rgba(maxima))]
K = CG_pos.iloc[:,:2].values
real_type = CG_pos.iloc[:,2].values
real_type_num = np.zeros(len(real_type))
for i in range(len(real_type)):
    if real_type[i] == 'DOPC':
        real_type_num[i] = 0
    if real_type[i] =='DPPC':
        real_type_num[i] = 1        
        
        
vor = Voronoi(K)
fig = voronoi_plot_2d(vor, show_vertices=False, point_size = 0.1, line_width=0.2)
for r in range(len(vor.point_region)):
    region = vor.regions[vor.point_region[r]]
    if not -1 in region:
        polygon = [vor.vertices[i] for i in region]
        plt.fill(*zip(*polygon), color=mapper.to_rgba(real_type_num[r]))
plt.xlim([0,2250]), plt.ylim([0,2250])
plt.xlabel('x'), plt.ylabel('y')
plt.legend(patch,["Disordered","Ordered"], bbox_to_anchor=(0.5, 1.1), ncol=2, loc='upper center')

fig.set_size_inches(15,15)
plt.show()
fig.savefig('output/real_large_protein.png',dpi=500)


ordered = dataset_whole[dataset_whole['Order']=='DPPC CHOL']
disordered = dataset_whole[dataset_whole['Order']=='DOPC CHOL']
un, cou = np.unique(ordered['Lipid Type'], return_counts=True)
un, cou = np.unique(disordered['Lipid Type'], return_counts=True)





