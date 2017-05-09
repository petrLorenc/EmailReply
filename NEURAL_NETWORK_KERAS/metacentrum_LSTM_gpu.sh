#!/bin/bash
# qsub -q gpu -l select=1:ncpus=8:ngpus=1:mem=12gb:scratch_local=10gb:cl_zubat=True -l walltime=24:00:00 metacentrum_LSTM_gpu.sh 
# sets home directory
DATADIR="/storage/praha1/home/petrlorenc2/keras"

module add python34-modules-gcc
module add python34-modules-intel

module add tensorflow-1.0.1-gpu-python3

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


cp $DATADIR/glove_training_LSTM.py ./glove_training_LSTM.py

python glove_training_LSTM.py

# moves the produced (output) data to user's home directory or leave it in SCRATCH if error occured 
cp ./train_model/* $DATADIR/train_model/ || export CLEAN_SCRATCH=false