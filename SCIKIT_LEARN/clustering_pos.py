import re
import numpy
import numpy as np

TEXT_DATA = "./data/train_5500.label.pos"
TAG_DATA = "./data/train_5500.label.tag"

texts = []  # list of text samples
mapping = {'ABBR' : 0, 
               'DESC' : 1,
               'ENTY' : 2,
               'HUM' : 3,
               'LOC' : 4,
               'NUM' : 5} # dictionary mapping label name to numeric id
# mapping = {'ABBR' : [1,0,0,0,0,0], 
#            'DESC' : [0,1,0,0,0,0],
#            'ENTY' : [0,0,1,0,0,0],
#            'HUM' : [0,0,0,1,0,0],
#            'LOC' : [0,0,0,0,1,0],
#            'NUM' : [0,0,0,0,0,1]}
labels = []  # list of label ids
split = 0.1

print('Processing text dataset')

with open(TEXT_DATA, mode="r", encoding="ISO-8859-1") as file:
        texts = file.readlines()

with open(TAG_DATA, mode="r", encoding="ISO-8859-1") as file:
        labels = [tag.strip() for tag in file.readlines()]


y_test = [mapping[y] for y in labels[:int(split*len(labels))] ]
x_test = [text.lower().strip() for text in texts[:int(split*len(texts))] ]

y_train = [mapping[y] for y in labels[int(split*len(labels)):] ]
x_train = [text.lower().strip() for text in texts[int(split*len(texts)):] ]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import classification_report

import pickle

from pprint import pprint
from time import time
import logging

classifiers = {
    # Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification)
    "MultinomialNB" : MultinomialNB(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    # Classifier implementing a vote among neighbors within a given radius
    "RadiusNeighborsClassifier": RadiusNeighborsClassifier(outlier_label=6),
    
    # Multi-layer Perceptron classifier.
    #"MLPClassifier":MLPClassifier(activation='relu', solver='sgd', learning_rate='adaptive'),
    
    # A random forest is a meta estimator that fits a number of decision tree classifiers 
    # on various sub-samples of the dataset and use averaging to improve the predictive 
    # accuracy and control over-fitting.
    "RandomForestClassifier": RandomForestClassifier(min_samples_leaf=10),
    
    # An AdaBoost [1] classifier is a meta-estimator that begins by fitting a classifier 
    # on the original dataset and then fits additional copies of the classifier on the same dataset 
    # but where the weights of incorrectly classified instances are adjusted such that subsequent 
    # classifiers focus more on difficult cases.
    "AdaBoostClassifier": AdaBoostClassifier(),
    "SGDClassifier": SGDClassifier()
} 

vectorizers = {
    "TfidfVectorizer" : TfidfVectorizer(),
    "CountVectorizer": CountVectorizer() # Bag of Words
}

best_scores = []

for name_classifier, classifier in classifiers.items():
    for name_vectorizer, vectorizer in vectorizers.items():
        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('clf', classifier) ])

        parameters = {
            'tfidf__stop_words': ({'english'}, None),
            'tfidf__max_features': (None, 5000, 10000, 50000),
            'tfidf__ngram_range': ((1, 1), (1, 2), (2, 2)),  # unigrams or bigrams
            'tfidf__lowercase': (True, False),
            'tfidf__max_df': (0.5, 1.0 , 2.0), # ignore terms that have a document frequency strictly higher than the given threshold
        }

        #If n_jobs was set to a value higher than one, the data is copied for each point in the grid (and not n_jobs times). This is done for efficiency reasons if individual jobs take very little time, but may raise errors if the dataset is large and not enough memory is available. 
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=8, verbose=1,scoring="accuracy")

        print("Performing grid search...")
        print("name_classifier:", name_classifier)
        print("name_classifier:", name_vectorizer)
        pprint(parameters)
        t0 = time()
        grid_search.fit(x_train, y_train)
        print("done in %0.3fs" % (time() - t0))
        print()

        best_scores.append((name_classifier, name_vectorizer, grid_search.best_score_, (time() - t0), grid_search.best_estimator_.get_params()))
        print("Best score: %0.3f" % grid_search.best_score_)
        
        predictions = grid_search.predict(x_test)
        print (classification_report(y_test, predictions))

print (best_scores)

import os
pickle.dump(best_scores, open(os.path.basename(__file__) + ".plk" , "wb"))

