#!/bin/bash
# sets home directory
DATADIR="/storage/praha1/home/petrlorenc2/keras"

module add python34-modules-gcc
module add python34-modules-intel

module add tensorflow-1.0.1-cpu-python3

# setup SCRATCH cleaning in case of an error
trap 'clean_scratch' TERM EXIT

# enters user's scratch directory
cd $SCRATCHDIR || exit 1

virtualenv keras_env
source keras_env/bin/activate

pip install keras
pip install h5py

mkdir glove_data
mkdir train_model
cp $DATADIR/glove_data/glove.6B.300d.txt ./glove_data/glove.6B.300d.txt
cp $DATADIR/train_5500.label.text ./train_5500.label.text
cp $DATADIR/train_5500.label.tag ./train_5500.label.tag


cp $DATADIR/glove_training_LSTM_CNN.py ./glove_training_LSTM_CNN.py

python glove_training_LSTM_CNN.py

# moves the produced (output) data to user's home directory or leave it in SCRATCH if error occured 
cp ./train_model/* $DATADIR/train_model/ || export CLEAN_SCRATCH=false