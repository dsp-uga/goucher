from Tiramisu import Tiramisu
from keras.callbacks import LearningRateScheduler
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, merge
from keras.regularizers import l2
from keras.models import Model
from keras import regularizers
import tensorflow as tf
from keras.backend import tensorflow_backend
from glob import glob
from numpy import array
from scipy.misc import imread


from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D

from keras import backend as K

from keras import callbacks
import math
import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from keras.layers.normalization import BatchNormalization

from keras.layers import Conv2D, Conv2DTranspose


config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)
K.set_image_dim_ordering('tf')

import cv2
import numpy as np
import json

class_weighting = [
 0.1826,
 4.5640,
 9.6446
]

def loadData(path):
    files = sorted(glob(path+'/frame*'))
    imgs = array([imread(f) for f in files])
    label = array(imread(path + '/mask.png'))
    return imgs, label


def change_size ( self , source , target_dim=[640,640] ):
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

imgs, label = loadData("/home/vamsi/Data Science Practicum/project4/FC-DenseNet/train/data/0b599d0670fcbafcaa8ed5567c0f4b10b959e6e49eed157be700bc62cffd1876")
train_data = array([change_size(self=" ", source=img,target_dim=[640,640]) for img in imgs])
train_label = change_size(self=" ", source=label,target_dim=[640,640])


train_label[train_label == 1] = 0
train_label[train_label == 2] = 1

'''
np.place(train_label,train_label<2,0)
np.place(train_label,train_label>0,2)
'''

train_label = (train_label == 1).astype(int)
print train_label

train_data = train_data.reshape(train_data.shape[0],train_data.shape[1],train_data.shape[2],1)
#print train_data.shape
train_labels = []
for i in range(100):
    train_labels.append(train_label)

train_labels = array(train_labels)
#train_labels[:train_data.shape[0],]=train_label
train_label = train_labels.reshape(100,train_label.shape[0],train_label.shape[1],1)



# load train data
#train_data = np.load('./data/train_data.npy')
#train_label = np.load('./data/train_label.npy')

#test_data = np.load('./data/val_data.npy')
#test_label = np.load('./data/val_label.npy')


layer_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
model = Tiramisu(layer_per_block)

#with open('./weights/prop_tiramisu_weights_67_12_func_10-e7_decay.best.hdf5') as model_file:
 #   model.load_weights(model_file, by_name=False)


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.00001
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)
optimizer = SGD(lr=0.01)
#optimizer = Adam(lr=1e-3, decay=0.995)

model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
TensorBoard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=5,
                                    write_graph=True, write_images=True)
filepath="weights/prop_tiramisu_weights_67_12_func_10-e7_decay.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2,
                             save_best_only=True, save_weights_only=False, mode='max')

callbacks_list = [checkpoint]

nb_epoch = 150
batch_size = 8

history = model.fit(train_data, train_label, batch_size=batch_size, epochs=nb_epoch,\
                    callbacks=callbacks_list, class_weight=class_weighting,verbose=1,\
                    validation_data=(train_data, train_label), shuffle=True)

model.save_weights('weights/prop_tiramisu_weights_67_12_func_10-e7_decay{}.hdf5'.format(nb_epoch))

import matplotlib.pyplot as plt
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


