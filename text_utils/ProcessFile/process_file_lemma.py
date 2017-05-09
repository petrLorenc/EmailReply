import re
import numpy
from nltk.corpus import wordnet
import nltk
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer


import argparse


#python -m nltk.downloader all

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def load_stop_words(path_to_file):
    with open(path_to_file, mode="r", encoding="ISO-8859-1") as file:
        lines = file.readlines()
    return [stopword.strip() for stopword in lines]
    
def get_lemmatized_words(sentence, stopwords):
    _temp = []
    tokens = nltk.word_tokenize(sentence)
    lmtzr = WordNetLemmatizer()
    
    for word, token in nltk.pos_tag(tokens):
        if word.strip() not in stopwords:
            _temp.append(lmtzr.lemmatize(word.strip().lower(), get_wordnet_pos(token)))
    return " ".join(_temp)


def load_file_to_process(path_to_file, path_to_stopwords):
    with open(path_to_file, mode="r") as file:
        lines = file.readlines()

    #nltk.download()
    input_x = []
    for line in lines:
        input_x.append(line)
    
    if path_to_stopwords != "":
      stopwords = load_stop_words(path_to_stopwords)
    else :
      stopwords = []

    for sentence in input_x:
      yield get_lemmatized_words(sentence, stopwords)

def main():
  parser = argparse.ArgumentParser(description='Description of your program')
  parser.add_argument('-i','--iFile', help='Input file', required=True)
  parser.add_argument('-o','--oFile', help='Output File', required=True)
  parser.add_argument('-s','--stopwordFile', default="" , help='Stopwords File', required=False)
  args = vars(parser.parse_args())
  inFile = args["iFile"]
  outFile = args["oFile"]
  #procesed = load_file_to_process(inFile, args["stopwordFile"])

  with open(outFile, mode="w") as file:
        for line in load_file_to_process(inFile, args["stopwordFile"]):
          file.write(str(line) + "\n")

if __name__ == "__main__":
    main()