import pickle 
import gensim

def read_corpus(unique_data, tokens_only=False):
    for i, line in enumerate(unique_data):
        if tokens_only:
            yield gensim.utils.simple_preprocess(line)
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

with open("./plain_kaggle_lemma.txt", mode="r",encoding="utf-8") as file:
	unique_data = file.readlines()

post_unique_data = list(read_corpus(unique_data))

#PV-DM is consistently better than PV-DBOW. PVDM
#alone can achieve results close to many results
#in this paper (see Table 2). For example, in IMDB,
#PV-DM only achieves 7.63%. The combination of
#PV-DM and PV-DBOW often work consistently better
#(7.42% in IMDB) and therefore recommended.
# https://groups.google.com/forum/#!msg/gensim/QuVMR8yso4s/bZ_naPZrAQAJ

# from Mikolov paper " optimal window size is 10 words."
model = gensim.models.doc2vec.Doc2Vec(size=500, dm=1, window=8, min_count=1, iter=500, workers=8)
model.build_vocab(post_unique_data)
model.train(post_unique_data, total_examples=model.corpus_count, epochs=model.iter)
model.save("doc2vec_kaggle_1000_iter.model")