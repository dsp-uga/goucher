# Project 4 Cilia Segmentation
## Team Goucher implementation of project 4 ( Cilia Segmentation ) of Data Science Practicum Spring 2018

### Project Description 

The task is to design an algorithm that learns how to segment cilia. Cilia are microscopic hairlike structures that protrude from literally every cell in your body. They beat in regular, rhythmic patterns to perform myriad tasks, from moving nutrients in to moving irritants out to amplifying cell-cell signaling pathways to generating calcium fluid flow in early cell differentiation. Cilia, and their beating patterns, are increasingly being implicated in a wide variety of syndromes that affected multiple organs.Connecting ciliary motion with clinical phenotypes is an extremely active area of research.

**Goal: find the cilia**


### Data Set

**Train/Test-** Data contains a bunch of folders (325 of them), named as hashes, each of which contains 100 consecutive frames of a gray-scale video of cilia. Masks contains a number of PNG images (211 of them), named as hashes (corresponding to the subfolders of data), that identify regions of the corresponding videos where cilia is.

Also within the parent folder are two text files: train.txt and test.txt. They contain the names, one per line, of the videos in each dataset. Correspondingly, there is masks in the masks folder for those named in train.txt; the others, we need to predict The training / testing split is 65 / 35, which equates to about 211 videos for training and 114 for testing.


### Requirements

The project requires the following technologies to be installed.
* Instructions to download and install Python can be found [here](https://www.python.org/).
* Instructions to download and install Keras can be found [here](https://keras.io/).
* Instructions to download and install Anaconda can be found [here](https://www.continuum.io/downloads).
* Instructions to download and install Tensor Flow can be found [here](https://www.tensorflow.org/install/install_mac).
* Instructions to download and install OpenCV Library can be found [here](https://opencv.org/).

Also see the [environment setup wiki page](https://github.com/dsp-uga/goucher/wiki/Environment-Setup) for more detailed installation instruction.
