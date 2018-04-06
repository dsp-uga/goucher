import gc
import cv2
import numpy as np
import json
import os
import math
import keras.models as models
import sys

from glob import glob
from numpy import array
from scipy.misc import imread

import tensorflow as tf
from Tiramisu import Tiramisu
from keras.callbacks import LearningRateScheduler
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, merge
from keras.regularizers import l2
from keras.models import Model
from keras import regularizers
from keras.backend import tensorflow_backend
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from keras import callbacks
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose


config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=23, inter_op_parallelism_threads=23))
tensorflow_backend.set_session(session)
K.set_image_dim_ordering('tf')

#Class weights to be applied 
class_weighting = [
 0.1826,
 4.5640,
 9.6446
]

def change_size ( source ):
        target_dim=[640,640]         
        """
        this function up samples the image into the given dimensions by padding the image with zeros (the background in the cilia images ) until it
        fits the given size.
        :param source: the array to resize
        :param target_dim:  the target size
        :return: returns the resized image
        """

        dims = source.shape
        if dims[0] != target_dim[0] or dims[1] != target_dim[1]:
            """
            if the source is not in the desired format change it to the desired size  
            """
            temp_mask = np.zeros((target_dim[0], target_dim[1]))

            temp_mask[:dims[0], :dims[1]] = source

            return temp_mask

        else:
            # if the image is already in the dessired size, return it.
            return source


def loadData(data_path):
    '''This function loads the data from the given path. This method was the 
    one from EveryOther class of preprocessing package but was redifined to make some changes.
    
    :param data_path: the path to load the data
    :return: returns the samples and labels as numpy arrays.'''
    samples = []
    masks = []
   
    for path, subdirs, frames in os.walk(data_path):
        for subdir in subdirs:
            #one in every five frames from the samples are loaded. The size of the frames are made to 640x640 
            #by padding in change_size method.
            t = [  change_size(cv2.imread(os.path.join(path, "%s/frame%04d.png" % (subdir, i)),0))  for i in range(0, 99, 5) ]
            t = [ np.expand_dims(x, axis=0)  for x in t ]
            gc.collect()


            #The label for the sample are loaded and are resized 640x640 just like the frames.
            y = change_size(cv2.imread( path+'/'+subdir+'/mask.png', 0))
            y= np.expand_dims( y, axis=0 )
            y=( y==2 ).astype(int)


            samples.extend(t) #extends all the frames of a sample to the samples variable
            for i in range( len(t)):#mask y is copied for len(t) times to even the samples and frames for the network.
                masks.append(y)

    return array(samples), array(masks) #converts the list of samples and masks into nparray.


data, label = loadData(sys.argv[1]) #imports the samples and masks from the given path.
gc.collect()
data_dims = data.shape
label_dims = label.shape

data = data.reshape(data_dims[0]*data_dims[1],data_dims[2],data_dims[3],1) #reshape to size(n_imgs, 640, 640, 1)
label = label.reshape(label_dims[0]*label_dims[1],label_dims[2],label_dims[3],1)#reshape to size(n_imgs, 640, 640, 1)

#Dividing the data to training and validation. 70 percent data for Training and 30 percent for validation.
train_data = data[:int(data_dims[0]*0.70)]
train_label = label[:int(label_dims[0]*0.70)]
val_data = data[int(data_dims[0]*0.70):]
val_label = label[int(label_dims[0]*0.70):]


def dice_coef(y_true, y_pred):
        """
        This is dice score implemetation
        :param y_true: ground truth
        :param y_pred: predicted
        :return: dice score calculated betweenthe actual and predicted versions
        """
        smooth = 1.0
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        print(K.max(y_true))

        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
        return dice_coef(y_true, y_pred)

#The number of layers for each dense block in the network.
layer_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
model = Tiramisu(layer_per_block)

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.00001
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)
optimizer = SGD(lr=0.01)
#optimizer = Adam(lr=1e-3, decay=0.995)

model.compile(loss=dice_coef_loss, optimizer=optimizer, metrics=["accuracy"])
TensorBoard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=10,
                                    write_graph=True, write_images=True)
filepath="weights/prop_tiramisu_weights_67_12_func_10-e7_decay.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
                             save_best_only=True, save_weights_only=False, mode='max')

callbacks_list = [checkpoint]

nb_epoch = 50
batch_size = 128

history = model.fit(train_data, train_label, batch_size=batch_size, epochs=nb_epoch,\
                    callbacks=callbacks_list, class_weight=class_weighting,verbose=1,\
                    validation_data=(val_data, val_label), shuffle=True)

model.save_weights('weights/prop_tiramisu_weights_67_12_func_10-e7_decay{}.hdf5'.format(nb_epoch))
