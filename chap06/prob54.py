import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train = pd.read_table('./data/train.feature.txt')
y_train = pd.read_table('./data/train.txt')

X_valid = pd.read_table('./data/valid.feature.txt')
y_valid = pd.read_table('./data/valid.txt')

model = joblib.load("./data/model_ch06.joblib")
train_pred = model.predict(X_train.values)
valid_pred = model.predict(X_valid.values)

print("train acc: {:.5f}".format(accuracy_score(y_train["CATEGORY"], train_pred)))
print("valid acc: {:.5f}".format(accuracy_score(y_valid["CATEGORY"], valid_pred)))