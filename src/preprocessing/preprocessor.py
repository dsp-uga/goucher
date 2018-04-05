"""Abstract class for preprocessors"""
"""
this class is based on the implementation at : 
https://github.com/dsp-uga/team-ball/blob/master/src/preprocessing/preprocessor.py
"""

import json
import numpy as np
import os

from numpy import array, zeros

class preprocessor:

    def __init__(self):
        self.data= None
        self.name = None
        self.x_train = None
        self.x_test = None
        self.y_train = None

    def loadSample(self, path):

        return  None,None, None;

    def preprocess (self):
        return None

    def load_from_files(self):
        """
        loads images from previously preprocessed files. 
        :return: a treplet of  ( train_x, train_y, test_dic )
        """

        if (not self.name is None and  not self.importPath is None):
            train_x = np.load(os.path.join(self.importPath,"x_train_"+ self.name + ".npy"))
            train_y = np.load(os.path.join(self.importPath, "y_train_"+ self.name + ".npy"))
            test_x = np.load(os.path.join(self.importPath, "x_test_"+ self.name + ".npy"))

            return train_x, train_y, test_x

        return None, None, None

    def save_to_file(self):
        """
        saves the preprocessed arays to files, files will be named base on the 
        preprocessor's name which should be set in init. 
        
        """

        if (not self.name is None and not self.importPath is None):
            np.savez_compressed(os.path.join(self.importPath, "x_train_" + self.name + ".npy"), self.x_train)
            np.savez_compressed(os.path.join(self.importPath, "y_train_" + self.name + ".npy") , self.y_train)
            # np.savez_compressed(os.path.join(self.importPath, "x_test_" + self.name + ".npy") , self.x_test)


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
            if (len(dims) == 2):

                temp_mask = np.zeros((target_dim[0], target_dim[1]))
            else:
                temp_mask = np.zeros((target_dim[0], target_dim[1],3))

            temp_mask[:dims[0], :dims[1]] = source

            return temp_mask

        else:
            # if the image is already in the dessired size, return it.
            return source
