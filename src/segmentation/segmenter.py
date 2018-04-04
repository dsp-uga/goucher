"""
this is a the abstract class for the segmenter
this is based on the script here :
https://github.com/dsp-uga/team-ball/blob/master/src/Classifiers/Classifier.py

"""
import os
import keras
from keras.models import load_model
import logging
import numpy as np


class Segmenter :

    def __init__(self):
        self.trained_model = None
        self.data = None
        self.classifier_name = None

        return

    def saveModel(self , export_path ):
        """
        this function is in charge of saving the model and it's weights
        :return:
        """
        if( self.trained_model ):
            self.trained_model.save( os.path.join( export_path ,  self.classifier_name+ ".h5"  ) )

        logging.info(  "Saved Model to : "+os.path.join( export_path ,  self.classifier_name+ ".h5" ))
        return

    def load_model(self , import_path):
        """
        loads model from file
        :param import_path:
        :return:  returns the model and sets it as the model in the class as well
        """
        self.trained_model = load_model(  os.path.join( import_path ,  self.classifier_name+ ".h5"  ))
        logging.info("Loaded Model at : " + os.path.join(import_path, self.classifier_name + ".h5"))
        return self.trained_model

    def train(self, x_train , y_train):
        """
        this funciton  trains the model which is defined in it's body and
        saves the model in the class for further prediction
        :return:
        """

        return self.trained_model

    def predict(self, data_dic, data_var = None ):
        """
        this function runs the prediction on the data
        :param data_dic:
        :return: the predicted values
        """
        if( self.classifier_name=='DUALINPUTUNET' ):
            temp = self.trained_model.predict([data_dic, data_var])
        else :
            temp = self.trained_model.predict(data_dic, batch_size=5)
        # print ( type( temp ) )
        # print( temp.shape )
        # print(np.max(temp) , np.mean( temp ), np.min( temp ) )
        temp = (temp>=0.5).astype(int)
        temp = np.sum(temp, axis=0)
        temp = (temp > 0.5).astype(int)
        # print ( temp.shape  , np.max( temp ), np.min(temp) )

        return  np.uint(temp) #
        # return np.uint(self.trained_model.predict(data_dic)[0])

        #
        # # X_test = X_test.reshape(X_test.shape + (1,))
        # ret={}
        # for item in data_dic:
        #
        #     temp =np.array([data_dic[item].reshape(data_dic[item].shape + (1,))])
        #     ret[item] =np.uint8( self.trained_model.predict(temp)[0].squeeze(axis=2) *255)
        # # ret = [{x: self.trained_model.predict()[0]} for x in data_dic]
        #
        # return ret