import re
import numpy
from nltk.corpus import wordnet
import nltk
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer

import argparse

#nltk.download("all")

# entity part
def extractEntities(ne_chunked):
    data = {}
    for entity in ne_chunked:
        if isinstance(entity, nltk.tree.Tree):
            text = " ".join([word for word, tag in entity.leaves()])
            ent = entity.label()
            data[text] = ent
        else:
            continue
    return data

def give_entities_in_sentence(sentence):
  tokens = nltk.word_tokenize(sentence)
  tagged = nltk.pos_tag(tokens)
  ne_chunked = nltk.ne_chunk(tagged, binary=False)
  return extractEntities(ne_chunked)

# lemma part
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
    
def get_lemmatized_words(sentence, stopwords, entity_in_sentence):
    _temp = []
    tokens = nltk.word_tokenize(sentence)
    lmtzr = WordNetLemmatizer()
    
    for word, token in nltk.pos_tag(tokens):
        if word.strip() not in stopwords:
            if word.strip() in entity_in_sentence:
              _temp.append(entity_in_sentence[word.strip()])
            else:
              _temp.append(lmtzr.lemmatize(word.strip().lower(), get_wordnet_pos(token)))
    return " ".join(_temp)

def load_file_to_process(path_to_file, path_to_stopwords):
    with open(path_to_file, mode="r", encoding="ISO-8859-1") as file:
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
      entity_in_sentence = give_entities_in_sentence(sentence)
      yield get_lemmatized_words(sentence, stopwords, entity_in_sentence)

def main():
  parser = argparse.ArgumentParser(description='Extract entites from sentences (per line) in file.')
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