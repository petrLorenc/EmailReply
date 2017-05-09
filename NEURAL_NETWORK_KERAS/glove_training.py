from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Embedding, LSTM, Dropout
from keras.models import Model


GLOVE_DIR = "./glove_data/glove.6B.50d.txt"
TEXT_DATA = "./train_5500.label.text"
TAG_DATA = "./train_5500.label.tag"

EMBEDDING_DIM = 50
MAX_SEQUENCE_LENGTH = EMBEDDING_DIM * 39 # longest sentences
MAX_NB_WORDS = 20000

VALIDATION_SPLIT = 0.2

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

embeddings_index = {}
f = open(GLOVE_DIR)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

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
x = LSTM(128, activation='relu')(embedded_sequences)
x = Dropout(0.5)(x)
x = LSTM(64, activation='relu')(x)
x = Dropout(0.5)(x)
x = LSTM(32, activation='relu')(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(32, activation='relu')(x)
preds = Dense(len(mapping2), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))