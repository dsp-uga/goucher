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


    def loadSample ( self, path ):
        """
        this funtion loads the images in the sample as one output image
        :param paht: path ot the sample this has to be in glob format describing path to TIFF images
        :return: returns one image which is the aggregated version of all images in the sample
        """

        # read image files
        files = sorted(glob(path))
        imgs = array([imread(f) for f in files])

        # merge files in to one image
        image = imgs.sum(axis=0)

        image = self.change_size( image, self.image_size )

        print(image.shape)

        return image


    def preprocess(self):
        """
        this funciton preopricess the imagaes into three arays, test_x tarin_x , train_y
        :return:
        """
        train_x = []  # None #np.array([])
        train_y = []  # None # np.array([])

        # create the trainig set
        if( not  self.trainingPath is None):
            for sample in sorted( os.listdir(self.trainingPath)) :
                images_glob_path = os.path.join( self.trainingPath,sample + "/*.png")
                mask_path = os.path.join( self.trainingPath, sample + '/mask.png')

                # load train_y
                y = self.change_size(cv2.imread( mask_path, 0))

                # take under account the skip count and lod the images
                for i in range(0, 99, self.skip_count):
                    temp_x= cv2.imread(os.path.join(self.trainingPath, "/%s/frame%04d.png" % (sample, i)))
                    train_x.append( self.change_size(temp_x))
                    train_y.append(y)


        # create the test set
        test_x = []
        if not self.testPath is None:
            for sample in sorted(os.listdir(self.testPath)):
                image = cv2.imread(os.path.join(self.testPath, "/%s/frame0050.png" % sample))
                test_x.append(image)

        train_x = array(train_x)
        train_y = array(train_y)
        test_x = array(test_x)

        self.x_train = train_x
        self.x_test = test_x
        self.y_train = train_y

        if( not self.exportPath is None):
            self.save_to_file()

        return train_x , train_y , test_x