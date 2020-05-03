import joblib
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from tqdm import tqdm

tqdm.pandas()

train_df = pd.read_table("./data/train.txt")
valid_df = pd.read_table("./data/valid.txt")
test_df = pd.read_table("./data/test.txt")

category2id = {
    "b":0,
    "t":1,
    "e":2,
    "m":3
}

train_df["CATEGORY_ID"] = train_df["CATEGORY"].map(category2id)
valid_df["CATEGORY_ID"] = valid_df["CATEGORY"].map(category2id)
test_df["CATEGORY_ID"] = test_df["CATEGORY"].map(category2id)

joblib.dump(train_df["CATEGORY_ID"].values, './data/y_train.joblib')
joblib.dump(valid_df["CATEGORY_ID"].values, './data/y_valid.joblib')
joblib.dump(test_df["CATEGORY_ID"].values, './data/y_test.joblib')

def sent2vec(text, model):
    vec = np.array([model[word] if word in model.vocab else np.zeros(shape=(model.vector_size,)) for word in text.split()])
    vec = np.mean(vec, axis=0)
    return vec

model = KeyedVectors.load_word2vec_format("./data/GoogleNews-vectors-negative300.bin", binary=True)

X_train = train_df["TITLE"].progress_map(lambda x: sent2vec(x, model))
X_valid = valid_df["TITLE"].progress_map(lambda x: sent2vec(x, model))
X_test = test_df["TITLE"].progress_map(lambda x: sent2vec(x, model))

joblib.dump(X_train.values, './data/X_train.joblib')
joblib.dump(X_valid.values, './data/X_valid.joblib')
joblib.dump(X_test.values, './data/X_test.joblib')