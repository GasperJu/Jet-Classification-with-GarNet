#import setGPU
import os
import h5py
import glob
import numpy as np
import matplotlib.pyplot as plt
from generatorGNN import DataGenerator
import sys
import numpy as np
from tensorflow.keras import metrics
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN

try:
    import tensorflow.keras as keras
except ImportError:
    import keras
K = keras.backend

import pandas as pd
from sklearn.utils import shuffle
from JetAdaptor import data_adaption_JetNet, percentage_info, lable_types, cut_groups_data
import matplotlib.pyplot as plt

print('Jet Classification with GarNet\n')

#characteristics
particles_number_per_jet = 30 
info_per_particle = 3
jet_types = 2
class_lable = ['Gluon', 'Top Quark']

print('Files...\n')
#files
gdata = "/home/gasper_ju/Documents/Física/IC/CERN/Diary/20*08*2021/g_jets.csv"
tdata = "/home/gasper_ju/Documents/Física/IC/CERN/Diary/20*08*2021/t_jets.csv"

print('Adapting Data...\n')
dtg = data_adaption_JetNet(particles_number_per_jet, gdata)
dtt = data_adaption_JetNet(particles_number_per_jet, tdata)

#jet_types dictionary starting from 1, since zero is none
dtg['Type'] = 1 #Gluon
dtt['Type'] = 2 #Top Quark

#join and mixing total data
dtall = pd.concat([dtg, dtt])
dtall = shuffle(dtall)
dtall.reset_index(drop=True, inplace=True)

#lable definition
lable = lable_types(dtall, class_lable)

#reshape input to Neural Network
dtall = np.reshape(dtall.astype(float).values, (lable.shape[0], particles_number_per_jet, info_per_particle))

print('Separating groups...\n')
#separate learning groups 
inputTrain, lableTrain, inputVal, lableVal, inputTest, lableTest =  cut_groups_data(dtall, lable)

print('Garnet...\n')

#Neural Network
from garnet import GarNetStack, GarNet

vmax = particles_number_per_jet 
quantize = False
x = keras.layers.Input(shape=(vmax, info_per_particle))
n = keras.layers.Input(shape=(1,), dtype='uint16') #connections graph
inputs = [x, n]

v = GarNet(16, 16, 16, simplified=True, collapse='mean', input_format='xn', 
                output_activation='relu', name='gar_1', quantize_transforms=quantize)([x, n])
v = keras.layers.Flatten()(v)
v = keras.layers.Dense(16, activation='relu')(v)
v = keras.layers.Dense(16, activation='relu')(v)
v = keras.layers.Dense(16, activation='relu')(v)
outLayer = keras.layers.Dense(jet_types, activation='softmax')(v)
    
model = keras.Model(inputs=inputs, outputs=outLayer)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

batch_size = 128 #CHANGE NUMBER OF BATCHS
n_epochs = 5

#graph nodes content keepers
V_train = np.ones((inputTrain.shape[0],1))*vmax
V_val = np.ones((inputVal.shape[0],1))*vmax
V_test = np.ones((inputTest.shape[0],1))*vmax

print('Learning...\n')
#learning process
history = model.fit((inputTrain, V_train), lableTrain, epochs=n_epochs,
                    validation_data = ((inputVal, V_val), lableVal), verbose=1,
                    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1),
                                 ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1),
                                 TerminateOnNaN()])

# plot training history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.yscale('log')
plt.title('Training History')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper right')
plt.savefig('Traininghistory.png', bbox_inches='tight')

print('Predictions:\n')

#prediction                                 
preds = model.predict((inputTest, V_test), batch_size=1000)

#conclusion
top, gluon, errgluon, errtop = percentage_info(preds, lableTest)


print('How many top quark jets were correctly identified?\n')
print("{:.2%}\n\n".format(top/len(preds)))

print('How many gluon jets were correctly identified?\n')
print("{:.2%}\n\n".format(gluon/len(preds)))

print('How many top quark jets were wrongly identified as gluon jets?\n')
print("{:.2%}\n\n".format(errgluon/len(preds)))

print('How many gluon jets were wrongly identified as top quark jets?\n')
print("{:.2%}\n\n".format(errtop/len(preds)))

import pandas as pd
from sklearn.metrics import roc_curve, auc
df = pd.DataFrame()
fpr = {}
tpr = {}
auc1 = {}

plt.figure()
for i, label in enumerate(class_lable):
        df[label] = lableTest[:,i]
        df[label + '_pred'] = preds[:,i]

        fpr[label], tpr[label], threshold = roc_curve(df[label],df[label+'_pred'])

        auc1[label] = auc(fpr[label], tpr[label])

        plt.plot(tpr[label],fpr[label],label='%s tagger, auc = %.1f%%'%(label,auc1[label]*100.))
plt.semilogy()
plt.xlabel("sig. efficiency")
plt.ylabel("bkg. mistag rate")
plt.ylim(0.000001,1)
plt.grid(True)
plt.legend(loc='lower right')
plt.savefig('Roc_Auc.png', bbox_inches='tight')
