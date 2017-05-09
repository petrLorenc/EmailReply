import re
import numpy
from nltk.corpus import wordnet
import nltk
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer

import argparse

#nltk.download("all")

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

def load_file_to_process(path_to_file):
    with open(path_to_file, mode="r", encoding="ISO-8859-1") as file:
        lines = file.readlines()

    #nltk.download()
    input_x = []
    for line in lines:
        input_x.append(line)

    for sentence in input_x:
      yield give_entities_in_sentence(sentence)

def main():
  parser = argparse.ArgumentParser(description='Extract entites from sentences (per line) in file.')
  parser.add_argument('-i','--iFile', help='Input file', required=True)
  parser.add_argument('-o','--oFile', help='Output File', required=True)
  args = vars(parser.parse_args())
  inFile = args["iFile"]
  outFile = args["oFile"]
  #procesed = load_file_to_process(inFile, args["stopwordFile"])

  with open(outFile, mode="w") as file:
        for line in load_file_to_process(inFile):
          file.write(str(line) + "\n")

if __name__ == "__main__":
    main()