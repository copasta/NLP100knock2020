import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train = pd.read_table('./data/train.feature.txt')
y_train = pd.read_table('./data/train.txt')

X_test = pd.read_table('./data/test.feature.txt')
y_test = pd.read_table('./data/test.txt')

model = joblib.load("./data/model_ch06.joblib")
train_pred = model.predict(X_train.values)
test_pred = model.predict(X_test.values)

print("train acc: {:.5f}".format(accuracy_score(y_train["CATEGORY"], train_pred)))
print("test acc: {:.5f}".format(accuracy_score(y_test["CATEGORY"], test_pred)))