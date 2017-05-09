import pickle 
import gensim

model = gensim.models.doc2vec.Doc2Vec.load("doc2vec.model")
print(model["how"][0])