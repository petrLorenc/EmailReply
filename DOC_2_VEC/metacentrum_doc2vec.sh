#!/bin/bash
# sets home directory
DATADIR="/storage/praha1/home/petrlorenc2/doc2vec_kaggle"

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

cp $DATADIR/doc2vec_kaggle.py ./doc2vec_kaggle.py

cp $DATADIR/plain_kaggle_lemma.txt ./plain_kaggle_lemma.txt

python doc2vec_kaggle.py

# moves the produced (output) data to user's home directory or leave it in SCRATCH if error occured 
cp ./doc2vec_kaggle_150_iter.model $DATADIR/doc2vec_kaggle_150_iter.model || export CLEAN_SCRATCH=false