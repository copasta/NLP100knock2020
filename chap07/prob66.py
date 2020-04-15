import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from tqdm import tqdm

def cos_sim(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return -1
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

model = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)

df = pd.read_csv('./data/wordsim353/combined.csv')
df["cos_sim"] = 0.0
for idx in tqdm(range(len(df)), total=len(df)):
    data = df.iloc[idx]
    w1 = data["Word 1"]
    w2 = data["Word 2"]
    sim_score = cos_sim(model[w1], model[w2])
    df["cos_sim"].iloc[idx] = sim_score

print(df[["Human (mean)", "cos_sim"]].corr(method="spearman"))