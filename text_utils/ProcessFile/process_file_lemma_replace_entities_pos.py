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

def get_pos(sentence, stopwords, to_text, entity_in_sentence):
  text_pos = nltk.pos_tag(nltk.word_tokenize(get_lemmatized_words(sentence, stopwords)))
  if to_text:
    return " ".join([str(word) if word.strip() not in entity_in_sentence else entity_in_sentence[word.strip()] 
      + " " + str(nltk.map_tag('en-ptb', 'universal', tag)) for word, tag in text_pos])
  else:
    simplifiedTags = [(word, nltk.map_tag('en-ptb', 'universal', tag)) for word, tag in text_pos]
    return simplifiedTags

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


def load_file_to_process(path_to_file, path_to_stopwords, to_text):
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
      yield get_pos(sentence, stopwords, to_text, entity_in_sentence)

def main():
  parser = argparse.ArgumentParser(description='Description of your program')
  parser.add_argument('-i','--iFile', help='Input file', required=True)
  parser.add_argument('-o','--oFile', help='Output File', required=True)
  parser.add_argument('-s','--stopwordFile', default="" , help='Stopwords File', required=False)
  parser.add_argument('-t','--toText', default=True ,type=bool , help='True/False if the output should be array or raw text', required=False)
  args = vars(parser.parse_args())
  inFile = args["iFile"]
  outFile = args["oFile"]
  #procesed = load_file_to_process(inFile, args["stopwordFile"])

  with open(outFile, mode="w") as file:
        for line in load_file_to_process(inFile, args["stopwordFile"], args["toText"]):
          file.write(str(line) + "\n")

if __name__ == "__main__":
    main()