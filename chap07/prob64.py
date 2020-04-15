from gensim.models import KeyedVectors
from tqdm import tqdm

model = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)

with open('./data/questions-words.txt', 'rt') as f, open('./data/prob64.txt', 'wt') as g:
    for line in tqdm(f.readlines(), total=len(f.readlines())):
        if line[0] == ":":
            g.write(line)
        else:
            line = line.split()
            result = list(model.most_similar(positive=[line[1], line[2]], negative=[line[0]])[0])
            sim_word = result[0]
            sim_score = result[1]
            g.write("{} {} {} {} {} {}\n".format(line[0], line[1], line[2], line[3], sim_word, sim_score))
