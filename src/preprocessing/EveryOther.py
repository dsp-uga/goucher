"""
this pre processor selects one every "skip_count" images in the sample 
for example if the sample has 100 images, out put will have 20 for skip_count =5 

test samples will have only one image chosen at random 
"""
from src.preprocessing.preprocessor import preprocessor
import numpy as np
import cv2
from numpy import array, zeros
import os
from glob import  glob
import cv2
import json


class EveryOther ( preprocessor ):

    def __init__(self, exportPath, trainingPath , testPath , images_size=[640,640], importPath = None , skip_count =5):
        self.exportPath = exportPath
        self.trainingPath = trainingPath
        self.testPath = testPath
        self.image_size = images_size
        self.importPath = importPath
        self.skip_count = skip_count
        self.name = "EveryOther_" + str(skip_count)
        self.x_test = None
        self.y_train = None
        self.x_train = None



    def preprocess(self):
        """
        this funciton preopricess the imagaes into three arays, test_x tarin_x , train_y
        :return:
        """
        train_x = [] # None #np.array([])
        train_y = []  # None # np.array([])

        # create the trainig set
        if( not  self.trainingPath is None):
            for sample in sorted(os.listdir(self.trainingPath)) :

                mask_path = os.path.join( self.trainingPath, sample + '/mask.png')

                # load train_y
                y = self.change_size(cv2.imread( mask_path, 0))
                y= np.expand_dims( y, axis=0 )
                y=( y==2 ).astype(int)

                # take under account the skip count and lod the images
                t = [  self.change_size(cv2.imread(os.path.join(self.trainingPath, "%s/frame%04d.png" % (sample, i)),0))  for i in range(0, 99, self.skip_count) ]
                t = [ np.expand_dims(x, axis=0)  for x in t ]
                train_x.extend(t)
                for i in range( len(t)):
                    train_y.append(y)

        # create the test set
        # test_x = []
        test_dic = {}
        test_size_ref = {}
        if not self.testPath is None:
            for sample in sorted(os.listdir(self.testPath)):
                image = cv2.imread(os.path.join(self.testPath, "%s/frame0050.png" % sample),0)
                test_size_ref[sample]= image.shape
                image = self.change_size(image)
                image = (image==2).astype(int).reshape(image.shape + (1,))
                test_dic[sample] = np.expand_dims(image, axis=0)
                # test_x.append(np.expand_dims(image, axis=0))

        train_x = np.vstack(train_x)
        train_y = np.vstack(train_y)
        # test_x = np.vstack(test_x)

        train_x = train_x.reshape(train_x.shape + (1,))
        train_y = train_y.reshape(train_y.shape + (1,))
        #test_x = test_x.reshape(test_x.shape + (1,))

        print(train_x.shape)
        print(train_y.shape)
        # print(test_x.shape)

        self.x_train = train_x
        # self.x_test = test_x
        self.y_train = train_y

        # if( not self.exportPath is None):
        #     self.save_to_file()

        return train_x , train_y , test_dic, test_size_ref