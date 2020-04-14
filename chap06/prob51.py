import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

X_train = pd.read_table('./data/train.txt')
X_valid = pd.read_table('./data/valid.txt')
X_test = pd.read_table('./data/test.txt')
df = pd.concat([X_train, X_valid, X_test])

tfv = TfidfVectorizer(
    analyzer='word', 
    token_pattern=r'(?u)\b\w+\b',
    ngram_range=(1, 1)
    )
tfv.fit(df["TITLE"])
dict_colums = {v:k for k, v in tfv.vocabulary_.items()}

feature_train = pd.DataFrame(tfv.transform(X_train["TITLE"]).toarray())
feature_train.columns = [dict_colums[col] for col in feature_train.columns]

feature_valid = pd.DataFrame(tfv.transform(X_valid["TITLE"]).toarray())
feature_valid.columns = [dict_colums[col] for col in feature_valid.columns]

feature_test = pd.DataFrame(tfv.transform(X_test["TITLE"]).toarray())
feature_test.columns = [dict_colums[col] for col in feature_test.columns]

print("train : {}, valid : {}, test : {}".format(feature_train.shape, feature_valid.shape, feature_test.shape))

feature_train.to_csv('./data/train.feature.txt', sep='\t', index=False)
feature_valid.to_csv('./data/valid.feature.txt', sep='\t', index=False)
feature_test.to_csv('./data/test.feature.txt', sep='\t', index=False)