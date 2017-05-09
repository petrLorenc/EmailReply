import re
import numpy
import numpy as np

#python -m nltk.downloader all

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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import svm

import pickle

from pprint import pprint
from time import time
import logging

SIZE_GLOVE_VECTOR = 300

# using coordinate-wise minimum - minimum from all first elements in vector and so on

TEXT_DATA = "./data/train_5500.label.text"
TAG_DATA = "./data/train_5500.label.tag"

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



def sentence_2_vec(model, sentence):
    vectors = []
    for word in sentence.split(): 
        if word in model:
            try:
                vectors.append(model[word.strip().lower()][0])
            except:
                pass
    #print(vectors)
    vectors = np.matrix(np.array(vectors).reshape(-1, SIZE_GLOVE_VECTOR))
    #print(vectors)
    maxVector = []
    minVector = []
    for i in range(SIZE_GLOVE_VECTOR):
        maxVector.append(np.max(vectors[:,i]))
        minVector.append(np.min(vectors[:,i]))
    return maxVector + minVector

def process_file(X_train, model):
    result_of_processing = []
    
    for sentence_in_X in X_train:
        vectors = sentence_2_vec(model, sentence=sentence_in_X)
        result_of_processing.append(vectors)
        
    return result_of_processing


## ---------------------------- testing classifier part -------------------------
def main():
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

    model = loadGloveModel("./glove_data/glove.6B." + str(SIZE_GLOVE_VECTOR) + "d.txt")

    classifiers = {
        "KNeighborsClassifier":KNeighborsClassifier(),
    
        # Classifier implementing a vote among neighbors within a given radius
        "RadiusNeighborsClassifier":RadiusNeighborsClassifier(outlier_label=6),
        
        # Multi-layer Perceptron classifier.
        #"MLPClassifier":MLPClassifier(activation='relu', solver='sgd', learning_rate='adaptive'),
        
        # A random forest is a meta estimator that fits a number of decision tree classifiers 
        # on various sub-samples of the dataset and use averaging to improve the predictive 
        # accuracy and control over-fitting.
        "RandomForestClassifier":RandomForestClassifier(min_samples_leaf=8),
        
        #An AdaBoost [1] classifier is a meta-estimator that begins by fitting a classifier 
        # on the original dataset and then fits additional copies of the classifier on the same dataset 
        # but where the weights of incorrectly classified instances are adjusted such that subsequent 
        # classifiers focus more on difficult cases.
        "AdaBoostClassifier":AdaBoostClassifier(),

        # The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset with more than a couple of 10000 samples.
        # Linear should be faster and it should be suitable for text data
        "SCV":svm.SVC(kernel='linear')
    } 

    best_scores = []


    post_X_train = np.array(process_file(x_train, model))
    #post_X_train = post_X_train.reshape(-1, SIZE_GLOVE_VECTOR)
    
    for name_classifier, classifier in classifiers.items():

        pipeline = Pipeline([
            ('clf', classifier) ])
        
        parameters = {}

        #If n_jobs was set to a value higher than one, the data is copied for each point in the grid (and not n_jobs times). This is done for efficiency reasons if individual jobs take very little time, but may raise errors if the dataset is large and not enough memory is available. 
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=8, verbose=1,scoring="accuracy")

        print("Performing grid search...")
        print("name_classifier:", name_classifier)
        pprint(parameters)
        t0 = time()
        grid_search.fit(post_X_train, y_train)
        print("done in %0.3fs" % (time() - t0))
        print()

        best_scores.append((name_classifier, grid_search.best_score_, (time() - t0), grid_search.best_estimator_.get_params()))
        print("Best score: %0.3f" % grid_search.best_score_)
        
        predictions = grid_search.predict(np.array(process_file(x_test, model)))
        print (classification_report(y_test, predictions))

    print (best_scores)

    import os
    pickle.dump(best_scores, open(os.path.basename(__file__) + ".plk" , "wb"))

    # END main --------------------


if __name__ == "__main__":
    main()
