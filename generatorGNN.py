import numpy as np
import tensorflow.keras as keras
import h5py
import glob
import random
from sklearn.utils import shuffle

class DataGenerator(keras.utils.Sequence):
    # 'Generates data for Keras'
    def __init__(self, label, fileList, batch_size, batch_per_file, vMax, verbose =0):
        'Initialization'
        self.verbose = verbose
        self.vMax = vMax
        self.label = label
        self.fileList = fileList
        self.batch_size = batch_size
        self.batch_per_file = batch_per_file
        self.myx = np.array([])
        self.myv = np.array([])
        self.myy = np.array([])
        # open first file
        self.f =  h5py.File(fileList[0],"r")
        self.X =  np.array(self.f.get('jetConstituentList'))
        self.X =  self.X[:,:,[5,7,10]]
        self.y = np.array(self.f.get('jets')[0:,-6:-1])
        self.X, self.y = shuffle(self.X, self.y)
        print(self.X.shape, self.y.shape, self.X.shape)
        self.nBatch = 0
        self.iFile = 0

#    def on_epoch_end(self):
#        print("%s boh" %self.label)

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        if self.verbose: print("%s LEN = %i" %(self.label, self.batch_per_file*len(self.fileList)))
        return self.batch_per_file*len(self.fileList)

    def __getitem__(self, index):
        if index == 0:
            # reshuffle data 
            if self.verbose: print("%s new epoch" %self.label)
            random.shuffle(self.fileList)
            self.iFile = 0
            self.nBatch = 0
            if self.verbose: print("%s new file" %self.label)
            if self.f != None: self.f.close()
            self.f = h5py.File(self.fileList[self.iFile], "r")
            self.X =  np.array(self.f.get('jetConstituentList'))
            self.X =  self.X[:,:,[5,7,10]]
            self.y = np.array(self.f.get('jets')[0:,-6:-1])
            self.X, self.y = shuffle(self.X, self.y)
        if self.verbose: print("%s: %i" %(self.label,index))

        #'Generate one batch of data'
        iStart = index*self.batch_size
        iStop = min(9999, (index+1)*self.batch_size)
        if iStop == 9999: iStart = iStop-self.batch_size
        self.myx = self.X[iStart:iStop,:,:]
        self.myv = np.ones((self.myx.shape[0],1))*self.vMax
        self.myy = self.y[iStart:iStop,:]
        if self.nBatch == self.batch_per_file-1:
            self.iFile+=1
            if self.iFile >= len(self.fileList):
                if self.verbose: print("%s Already went through all files" %self.label)
            else:
                if self.verbose: print("%s new file" %self.label)
                self.f.close()
                self.f = h5py.File(self.fileList[self.iFile], "r")
                self.X =  np.array(self.f.get('jetConstituentList'))
                self.X =  self.X[:,:,[5,7,10]]
                self.y = np.array(self.f.get('jets')[0:,-6:-1])
                self.X, self.y = shuffle(self.X, self.y)
            self.nBatch = 0
        else:
            self.nBatch += 1
        return [self.myx, self.myv], self.myy
