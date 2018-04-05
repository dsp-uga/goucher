"""
this preprocessor, adds optical flow information to the input sequence to the
netwwork.
this works based on two ideas, an overall flow and a per 5 frame flow calculation
"""
from src.preprocessing.preprocessor import preprocessor
import numpy as np
import cv2
from numpy import array, zeros
import os
from glob import  glob
import cv2
import json


class OpticalFlow ( preprocessor ):

    def __init__(self, exportPath, trainingPath , testPath , images_size=[640,640], importPath = None , skip_count =5):
        self.exportPath = exportPath
        self.trainingPath = trainingPath
        self.testPath = testPath
        self.image_size = images_size
        self.importPath = importPath
        self.skip_count = skip_count
        self.name = "Optical_Flow_" + str(skip_count)
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
        train_vars = []
        train_of = []
        # create the trainig set
        if( not  self.trainingPath is None):
            for sample in sorted(os.listdir(self.trainingPath)) :

                mask_path = os.path.join( self.trainingPath, sample + '/mask.png')


                if  os.path.exists(os.path.join( self.trainingPath, sample + '/OpticalFlow.png')  ):
                    the_of = cv2.imread( os.path.join( self.trainingPath, sample + '/OpticalFlow.png') , 1 )
                else:
                    frame1 = self.cv_resize( cv2.imread( os.path.join( self.trainingPath, sample + '/frame0001.png')))
                    frame2 = self.cv_resize( cv2.imread( os.path.join( self.trainingPath, sample + '/frame0050.png')))
                    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                    hsv = np.zeros_like(frame1)
                    hsv[..., 1] = 255

                    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    hsv[..., 0] = ang * 180 / np.pi / 2
                    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    cv2.imwrite( os.path.join( self.trainingPath, sample + '/OpticalFlow.png'), bgr )

                    the_of = bgr

                the_of = np.expand_dims(the_of,
                                            axis=0)  # np.expand_dims(the_of.reshape(the_of.shape + (1,)), axis=0) #   np.expand_dims(the_of, axis=0)

                # make varinaces
                if  os.path.exists(os.path.join( self.trainingPath, sample + '/basicVariance.png')  ):
                    the_var = cv2.imread( os.path.join( self.trainingPath, sample + '/basicVariance.png') ,0 )
                else:
                    files = sorted( glob( os.path.join(self.trainingPath, "%s/frame*.png" % sample) ) )
                    files = np.array([self.change_size(cv2.imread(x, 0)) for x in files])
                    variances = np.var(files, axis=0)
                    variances = (variances / np.max(variances)) * 255

                    del (files )

                    cv2.imwrite( os.path.join( self.trainingPath, sample + '/basicVariance.png'), variances )

                    the_var = variances

                the_var = np.expand_dims(the_var.reshape(the_var.shape + (1,)), axis=0)
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
                    train_vars.append(the_var)
                    train_of.append(the_of)

        # create the test set
        # test_x = []
        test_dic = {}
        test_size_ref = {}
        test_vars = {}
        tesr_ofs= {}
        if not self.testPath is None:
            for sample in sorted(os.listdir(self.testPath)):
                # image = cv2.imread(os.path.join(self.testPath, "%s/frame0050.png" % sample),0) #/ 255
                # test_size_ref[sample]= image.shape
                # image = self.change_size(image)
                # image = image.reshape(image.shape + (1,))
                # test_dic[sample] = np.expand_dims(image, axis=0)
                print (os.path.join(self.testPath, "%s/frame%04d.png" % (sample, i)))
                if  '.DS_Store' in sample : continue
                the_var= None

                if  os.path.exists(os.path.join( self.testPath, sample + '/OpticalFlow.png')  ):
                    the_of = cv2.imread( os.path.join( self.testPath, sample + '/OpticalFlow.png') ,1)
                else:
                    frame1 = self.cv_resize( cv2.imread( os.path.join( self.testPath, sample + '/frame0001.png')))
                    frame2 = self.cv_resize( cv2.imread( os.path.join( self.testPath, sample + '/frame0050.png')))
                    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                    hsv = np.zeros_like(frame1)
                    hsv[..., 1] = 255

                    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    hsv[..., 0] = ang * 180 / np.pi / 2
                    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    cv2.imwrite( os.path.join( self.testPath, sample + '/OpticalFlow.png'), bgr )

                    the_of = bgr

                the_of =   np.expand_dims(the_of, axis=0) # np.expand_dims(the_of.reshape(the_of.shape + (1,)), axis=0) #   np.expand_dims(the_of, axis=0)


                # make varinaces
                if  os.path.exists(os.path.join(self.testPath, sample + '/basicVariance.png')):
                    the_var = cv2.imread(os.path.join(self.testPath, sample + '/basicVariance.png'), 0)
                else:
                    files = sorted(glob(os.path.join(self.testPath, "%s/frame*.png" % sample)))
                    files = np.array([self.change_size(cv2.imread(x, 0)) for x in files])
                    variances = np.var(files, axis=0)
                    variances = (variances / np.max(variances)) * 255
                    del (files )
                    cv2.imwrite(os.path.join(self.testPath, sample + '/basicVariance.png'), variances)

                    the_var = variances

                the_var = np.expand_dims(the_var.reshape(the_var.shape + (1,)), axis=0)
                test_vars[sample] = the_var

                t = [cv2.imread(os.path.join(self.testPath, "%s/frame%04d.png" % (sample, i)), 0)
                     for i in range(0, 99, 25)]

                test_size_ref[sample] = t[0].shape

                t = [ self.change_size(x) for x in  t ]

                t = [np.expand_dims(x.reshape(x.shape + (1,)), axis=0)  for x in t]

                temp_vars  = []
                temp_ofs = []
                for i in range ( len(t) ):
                    temp_vars.append(the_var)
                    temp_ofs.append( the_of )

                test_vars[sample] = np.vstack(temp_vars)
                tesr_ofs[sample] = np.vstack( temp_ofs )

                test_dic[sample] = np.vstack(t)


                # test_x.append(np.expand_dims(image, axis=0))

        train_x = np.vstack(train_x)
        train_y = np.vstack(train_y)
        train_vars = np.vstack(train_vars)
        train_of = np.vstack( train_of )
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

        return train_x , train_y , test_dic, test_size_ref, train_vars, test_vars, train_of , tesr_ofs

    def cv_resize (self, im):
        """
        code from : https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
        :return: 
        """
        desired_size = 640
        old_size = im.shape[:2]  # old_size is in (height, width) format
        ratio = desired_size/ max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # new_size should be in (width, height) format
        im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)
        return new_im