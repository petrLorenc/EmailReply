import pickle 
import gensim

def read_corpus(unique_data, tokens_only=False):
    for i, line in enumerate(unique_data):
        if tokens_only:
            yield gensim.utils.simple_preprocess(line)
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

data = pickle.load(open("enron_unique_questions.plk", mode="rb"))

unique_data = list(set(data))
post_unique_data = list(read_corpus(unique_data))

model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=150)
model.build_vocab(post_unique_data)
model.train(post_unique_data, total_examples=model.corpus_count, epochs=model.iter)
model.save("doc2vec.model")