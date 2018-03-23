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

# create folders
if [ ! -d "train" ]
then
    mkdir "train"
fi

if [ ! -d "test" ]
then
    mkdir "test"
fi

echo "staring to download training"
cd test

while read -r line || [[ -n "$line" ]]; do
    echo "$line"
    wget "https://storage.googleapis.com/uga-dsp/project4/data/$line.tar"
    tar -xf  "$line.tar"
    rm "$line.tar"
done < "../test.txt"


echo "downloading train"

cd ../train

while read -r line || [[ -n "$line" ]]; do
    echo "$line"
    wget "https://storage.googleapis.com/uga-dsp/project4/data/$line.tar"
    tar -xf  "$line.tar"
    rm "$line.tar"
done < "../train.txt"
