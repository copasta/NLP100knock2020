from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
for result in model.most_similar(u"United_States", topn=10):
    print(result)