import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score

X_test = pd.read_table('./data/test.feature.txt')
y_test = pd.read_table('./data/test.txt')

model = joblib.load("./data/model_ch06.joblib")
test_pred = model.predict(X_test.values)

print("---NONE---")
print("test recall : {}".format(recall_score(y_test["CATEGORY"], test_pred, average=None)))
print("test precision : {}".format(precision_score(y_test["CATEGORY"], test_pred, average=None)))
print("test f1-score : {}".format(f1_score(y_test["CATEGORY"], test_pred, average=None)))
print("---MICRO---")
print("test recall : {:.5f}".format(recall_score(y_test["CATEGORY"], test_pred, average="micro")))
print("test precision : {:.5f}".format(precision_score(y_test["CATEGORY"], test_pred, average="micro")))
print("test f1-score : {:.5f}".format(f1_score(y_test["CATEGORY"], test_pred, average="micro")))
print("---MACRO---")
print("test recall : {:.5f}".format(recall_score(y_test["CATEGORY"], test_pred, average="macro")))
print("test precision : {:.5f}".format(precision_score(y_test["CATEGORY"], test_pred, average="macro")))
print("test f1-score : {:.5f}".format(f1_score(y_test["CATEGORY"], test_pred, average="macro")))