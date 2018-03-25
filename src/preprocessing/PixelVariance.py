import numpy as np
from glob import glob
from numpy import array
from scipy.misc import imread
import scipy.misc
import sys
import os

'''
This method takes 3d image sequence as numpy array argument. 
The image sequence is then converted to a 2-d matrix with 
pixels as rows and pixel values for images as columns. 
Then variance of the pixels are calculated and stored in a 
variable. The top variant pixels as replaced with a given 
pixel value. 
'''
def highVariance(imgs,hv=1,pix=125):
    timgs = transform(imgs)
    varimg = timgs.var(1)
    sortvar = sorted(varimg,reverse=True)
    sortvar = sortvar[:hv]
    a = np.full((1,int(imgs.shape[0])),pix)
    for mv in sortvar:
        timgs[int(np.where(varimg == mv)[0][0])] = a
    return timgs

'''
Tranforms the given image into a 2d matrix
'''
def transform(imgs):
    t_imgs = np.transpose(imgs)
    tod_data = t_imgs.reshape(imgs.shape[1]*imgs.shape[2], imgs.shape[0])
    return tod_data

'''
Saves the modified image matrices to a image. 
This function contains the arguments of the 
'path' where the image will be saved 
'varimgs' the modified image matrices
'imgs' orginal image matrix 
'''
def saveImage(path, varimgs, imgs):
    timgs = np.transpose(varimgs)
    rimgs = timgs.reshape(imgs.shape[0],imgs.shape[1],imgs.shape[2])
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range (0,len(rimgs)):
        scipy.misc.imsave(path+'/img'+str(i)+'.png', rimgs[i])
'''
Function to load the images from the given path
'''
def loadImgs(path):
    files = sorted(glob(path+'/frame*'))
    imgs = array([imread(f) for f in files])
    return imgs


if __name__ == "__main__":
    path = sys.argv[1]
    imgs = loadImgs(path)
    varImgs = highVariance(imgs)
    saveImage(path+"/preprocess",varImgs,imgs)
