import pandas as pd
from sklearn.model_selection import train_test_split

corpus_df = pd.read_csv("data/NewsAggregatorDataset/newsCorpora.csv", sep='\t', header=None)
corpus_df.columns = ["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"]
use_publisher = ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]
corpus_df = corpus_df[corpus_df["PUBLISHER"].map(lambda x:x in use_publisher)].sample(frac=1.0, random_state=1234)

X = corpus_df[["TITLE", "CATEGORY"]]

X_train, X_test = train_test_split(X, test_size=0.2, random_state=1234, stratify=X["CATEGORY"])
X_valid, X_test = train_test_split(X_test, test_size=0.5, random_state=1234, stratify=X_test["CATEGORY"])
print("train : {}, valid : {}, test : {}".format(X_train.shape, X_valid.shape, X_test.shape))

X_train.to_csv('./data/train.txt', sep='\t', index=False)
X_valid.to_csv('./data/valid.txt', sep='\t', index=False)
X_test.to_csv('./data/test.txt', sep='\t', index=False)

print("--TRAIN--")
print(X_train["CATEGORY"].value_counts())
print("--VALID--")
print(X_valid["CATEGORY"].value_counts())
print("--TEST--")
print(X_test["CATEGORY"].value_counts())