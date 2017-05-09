from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, Dropout
from keras.models import Model
from keras import regularizers
from keras.callbacks import ModelCheckpoint



#pip install h5py
EMBEDDING_DIM = 300
EPOCH = 100
BATCH_SIZE = 32
LONGEST_SENTENCE = 39

GLOVE_DIR = "./glove_data/glove.6B." + str(EMBEDDING_DIM) + "d.txt"
#GLOVE_DIR = "../QuestionClassificationScikit/glove_data/glove.6B." + str(EMBEDDING_DIM) + "d.txt"

TEXT_DATA = "./train_5500.label.text"
TAG_DATA = "./train_5500.label.tag"
SAVE_DIR = "./train_model"
NAME_MODEL = "model_LSTM_CNN"


#MAX_SEQUENCE_LENGTH = EMBEDDING_DIM * 39 # longest sentences
MAX_SEQUENCE_LENGTH = LONGEST_SENTENCE # because of memory limit on metacentrum

MAX_NB_WORDS = 20000

VALIDATION_SPLIT = 0.1

# first, build index mapping words in the embeddings set
# to their embedding vector
texts = []  # list of text samples
mapping2 = {'ABBR' : 0, 
               'DESC' : 1,
               'ENTY' : 2,
               'HUM' : 3,
               'LOC' : 4,
               'NUM' : 5} # dictionary mapping label name to numeric id
labels = []  # list of label ids


print('Indexing word vectors.')

# embeddings_index = {}
# f = open(GLOVE_DIR, mode="r", encoding="ISO-8859-1")
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()

def loadGloveModel(gloveFile):
        print ("Loading Glove Model")
        f = open(gloveFile,'r', encoding="UTF-8")
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = [float(val) for val in splitLine[1:]]
            model[word] = np.array(embedding).reshape(1,-1)
        print ("Done.",len(model)," words loaded!")
        return model

embeddings_index = loadGloveModel(GLOVE_DIR)

# second, prepare text samples and their labels
print('Processing text dataset')

with open(TEXT_DATA, mode="r", encoding="ISO-8859-1") as file:
        texts = file.readlines()

with open(TAG_DATA, mode="r", encoding="ISO-8859-1") as file:
        labels = [mapping2[tag.strip()] for tag in file.readlines()]

print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
# return_sequences=True if the next layer is also time dependanat
# Conv1D
# 1 - filters: Integer, the dimensionality of the output space (i.e. the number output of filters in the convolution).
# 2 - kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
# 3 - strides: An integer or tuple/list of a single integer, specifying the stride length of the convolution. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
# MaxPooling1D
# 1 - pool_size: Integer, size of the max pooling windows.
# 2 - strides: Integer, or None. Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size.
# train a 1D convnet with global maxpooling
x = Dropout(0.25)(embedded_sequences)
x = Conv1D(128, 25, padding='valid', activation='relu', strides=1)(x)
x = MaxPooling1D(pool_size=5)(x)
x = LSTM(128, activation='relu', dropout=0.5, return_sequences=False)(x) # dropout original 0.5
x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
preds = Dense(len(mapping2), activation='softmax')(x)

# This creates a model that includes
# the sequence_input and preds + all between them
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['mae', 'acc'])

# Save the model as png file
# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='model_LSTM_CNN.png', show_shapes=True)

# checkpoint
filepath= SAVE_DIR + "/weights-" + NAME_MODEL + "-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCH,
          validation_data=(x_val, y_val),
          callbacks=callbacks_list)


# serialize model to JSON
model_json = model.to_json()
with open(SAVE_DIR + NAME_MODEL+ ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(SAVE_DIR + NAME_MODEL + ".h5")
print("Saved model to disk, to folder " , SAVE_DIR)

# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")

# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))