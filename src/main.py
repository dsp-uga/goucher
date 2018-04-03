"""
this file contains the main runner for the project
"""

import argparse
import sys
import os
import logging
from src.preprocessing import preprocessor
from src.preprocessing import EveryOther
from src.segmentation.segmenter import Segmenter
from src.segmentation.UnetSegmenter import UNET_Segmenter
from src.segmentation.DualInputUNETSegmenter import Dual_Input_UNET_Segmenter
from src.preprocessing.BasicVariance import BasicVariance
from src.postprocessing.Postprocessor import  postProcess
description = ' '

parser = argparse.ArgumentParser(description=description, add_help='How to use', prog='python main.py <options>')

parser.add_argument("-d", "--dataset", default="../data/tarin/",
                    help='Path to the training data [DEFAULT: "data/tarin/"]')

parser.add_argument("-ts", "--testset", default=None,
                    help='Path to the testing data [DEFAULT: None]')

parser.add_argument("-m", "--model", default="unet",
                    help='model to be used in the segmentation can be UNET/FCN/NMF [DEFAULT: "FCN"]')

parser.add_argument("-t", "--train", action="store_true",
                    help='To ensure a model is being trained')

parser.add_argument("-p", "--predict", action="store_true",
                    help='To ensure a segmentation is performed on the test set (This requires --testset to have value)')

parser.add_argument("-e", "--epoch", default="1024",
                    help='Sets number of epochs for which the network will be trained')

parser.add_argument("-b", "--batch", default="4",
                    help='sets the batch size for training the models')

parser.add_argument("-pp", "--preprocessor", default="everyother",
                    help='Chooses the Preprcessor to be applied ')

parser.add_argument("-ep", "--exportpath", default=None,

                    help='Chooses the path to export model and numpy files')

parser.add_argument("-o", "--output", default=None,

                    help='sets the path for the output files to be stored')


parser.add_argument("-lf", "--logfile", default="log.log",
                    help="Path to the log file, this file will contain the log records")


# compile arguments
args = parser.parse_args()

# setup logging
logging.basicConfig(filename=args.logfile, level=logging.INFO, filemode="w",
                    format=" %(asctime)s - %(module)s.%(funcName)s - %(levelname)s : %(message)s ")

the_preprocessor = None
the_Segmenter = None

# set the preprocessor
if (args.preprocessor == "everyother"):
    the_preprocessor = EveryOther.EveryOther(images_size=[640, 640],trainingPath=args.dataset, testPath=args.testset,
                                         exportPath=args.exportpath, importPath=args.exportpath )
elif (args.preprocessor== 'basicvar'):
    the_preprocessor = BasicVariance(images_size=[640, 640],trainingPath=args.dataset, testPath=args.testset,
                                         exportPath=args.exportpath, importPath=args.exportpath, skip_count=25 )
else:
    the_preprocessor = preprocessor.preprocessor()

# set the set classifier :

if args.model == "unet":
    the_Segmenter = UNET_Segmenter()
if args.model == 'dualinpuunet':
    the_Segmenter = Dual_Input_UNET_Segmenter()
else:
    the_Segmenter = Segmenter()

# -------------- Loading the data

# try to load pre calculated data :
# try:
#     x_train, y_train, x_test, test_size_ref = the_preprocessor.load_from_files()
#
# except FileNotFoundError:
    # if there is no file to load set them as null, they will be loaded autiomatically
x_train, y_train, x_test, test_size_ref = None, None, None, None

# check if there is no data, read them from input ( this will take time! )
if  ( x_train is None):
    logging.info("Loading data from original data")
    x_train, y_train, x_test, test_size_ref , train_var, test_vars = the_preprocessor.preprocess()
    logging.info("Done loading data from original data")
else:
    logging.info("data loaded from pre-calculated files")
# --------------- Loading the data


# --------------- train model!
if( args.train ):
    logging.info("Starting training")
    if (args.model == 'unet'):
        model = the_Segmenter.train(x_train=x_train, y_train=y_train, epochs=int(args.epoch) ,batch_size=int(args.batch) )
    else:
        model = the_Segmenter.train(x_train=x_train, y_train=y_train, v_train=train_var, epochs=int(args.epoch),
                                    batch_size=int(args.batch))
    the_Segmenter.saveModel(args.exportpath)
    logging.info("Done with training")
else:
    model = the_Segmenter.load_model(args.exportpath)

# --------------- train model!

#------------ predict
if( args.predict  and x_test ):
    # run the prediction
    predicted={}
    import numpy as np
    for key in x_test :
        print( x_test[key].shape , np.max( x_test[key]) , np.min( x_test[key] )  )
        predicted[key] = the_Segmenter.predict( x_test[key] )


    # save the results
    postProcess(theDic=predicted,output_path=args.output , size_dic=test_size_ref)