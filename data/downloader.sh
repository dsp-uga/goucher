#!/usr/bin/env bash

# download the guide files if test and train are
if [ ! -f "test.txt"   ]
then
    wget "https://storage.googleapis.com/uga-dsp/project4/test.txt"
fi
if [ ! -f "train.txt"   ]
then
    wget "https://storage.googleapis.com/uga-dsp/project4/train.txt"
fi


