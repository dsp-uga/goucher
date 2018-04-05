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

### Approach

* #### UNET
    U-Net is convolutional network architecture for fast and precise segmentation of images.This deep neural network is implemented with    Keras functional API, which makes it extremely easy to experiment with different interesting architectures.

  In first attempt we just trained the network with normal images extracted form the frame.

  In Second attempt variance was added as a one more input in U-NET

  and Finally we also added the Optical Flow form Opencv as the third input in U-NET.

  Thus,

  #### Input for U-NET=Images+Variance+Optical-flow
  To know more about variance go-to wiki preprocessing page or click [here](https://github.com/dsp-uga/goucher/wiki/Pre-Processing)
  
* #### Fast RCN

  Fast R-CNN builds on previous     work to efficiently classify object proposals using deep convolutional networks. Compared to  previous work, Fast R-CNN employs several innovations to improve training and testing speed while also increasing detection accuracy. Fast R-CNN  trains the very deep      VGG16 network 9x faster than R-CNN.

  Unfortunately this model did not work well and we continued implementing with U-NET.
  More about the model is added in experiment-rcn branch

* #### The One Hundred Layers Tiramisu:

The typical segmentation architecture is composed of a downsampling path responsible for extracting coarse semantic features, followed by an upsampling path trained to recover the input image resolution at the output of the model and, optionally, a post-processing module (e.g. Conditional Random Fields) to refine the model predictions. We extend DenseNets to deal with the problem of semantic segmentation.

 Althoug this network required GPU and it took too much time for processing when compare to U-NET. 

### Downloader

There is a per-written bash script to download the data-set locally.

File 'downloader.sh' is a bash script to download files from the google storage.
This can be found [here](https://github.com/dsp-uga/goucher/blob/Experimental-RCN/data/downloader.sh) in the repository.
To run the downloader use the following command in your local directory :

`$bash downloader.sh`
  
 ### Execution
Main Program

  ` src/main.py is the main file to run the project`
  
Following keys are settable through command arguments :

* `--epoch` : sets number of epochs for which training will go ( this is applicable to UNET and R-FCNN models )

* `--dataset` : this sets the path to the training files. target folder should contain one folder per sample and they have to comply to the original dataset format
* `--testset` : this is the path test samples, this folder should contain one folder per each sample and they should follow the original dataset's format
* `--model` : this sets the model to either of UNET/FCN/tiramisu
* `--train` : if supplied training will be done
* `--exportpath` : set the path to which numpy arrays for train and test file as well as model will be saved. note that this same path will be used to load them
* `--predict` : if supplied, prediction will also be done on the data set
* `--logfile` : sets the path to the logging file
* `--preprocessor` : selects the preprocessor to be applied to the dataset
* `--batch` : sets the batch size for the training models, this only applies to UNET and FCN

Sample ruuning command

`$ python main.py --train --predict --exportpath="../output" --dataset="../data/train" --testset="../data/test"`

### Final Output

We predit the cilia region in form of an image as shown

<p align="left">
  <img width="300" height="200" src="https://github.com/dsp-uga/goucher/blob/Experimental-RCN/analysis/initial%20image">
  <img width="300" height="200" src="https://github.com/dsp-uga/goucher/blob/Experimental-RCN/analysis/final%20image">
</p>

where

**LEFT:** A single frame from the grayscale video. 

**RIGHT:** A 3-label segmentation of the video 

For which label 2 is cilia.

The result for testing dataset were submitted on AutoLab.

**The overalll accuracy range achieved is around 39-44 % depening uopn batch size, kernel, epoch etc.**

### Contributors

See [contributor](https://github.com/dsp-uga/goucher/blob/master/CONTRIBUTORS.md) file for more details.

### License

This project is licensed under the MIT License - see the [License file](https://github.com/dsp-uga/goucher/blob/master/LICENSE) for details

### Acknowledgments

* This project was completed as a part of the Data Science Practicum 2018 course at the University of Georgia
* Dr. Shannon Quinn is responsible for the problem formulation and initial guidance towards solution methods.
* We partly used provided code from This Repository. It provides implementation of Fully Convolutional Networks with Keras
Other resources used have been cited in their corresponding wiki page.

### References

* https://arxiv.org/pdf/1707.06314.pdf- Research paper for U-net
* https://github.com/dsp-uga/sp18/tree/master/projects/p4
* https://github.com/yhenon/keras-frcnn
* https://arxiv.org/abs/1504.08083

