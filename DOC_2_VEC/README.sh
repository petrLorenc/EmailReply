#!/bin/bash
# sets home directory
DATADIR="/storage/praha1/home/petrlorenc2"

module add python34-modules-gcc
module add python34-modules-intel

module add tensorflow-1.0.1-cpu-python3

virtualenv gensim_env
source gensim_env/bin/activate

pip install gensim


# setup SCRATCH cleaning in case of an error
trap 'clean_scratch' TERM EXIT

# enters user's scratch directory
cd $SCRATCHDIR || exit 1

cp $DATADIR/doc2vec.py ./doc2vec.py

cp $DATADIR/enron_unique_questions.plk ./enron_unique_questions.plk

python doc2vec.py

# moves the produced (output) data to user's home directory or leave it in SCRATCH if error occured 
cp ./doc2vec.model $DATADIR/doc2vec.model || export CLEAN_SCRATCH=false