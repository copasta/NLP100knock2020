import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

DEBUG = False
if DEBUG:
    nrows=50
else:
    nrows=None
X_train = pd.read_table('./data/train.feature.txt', nrows=nrows)
y_train = pd.read_table('./data/train.txt', nrows=nrows)

X_test = pd.read_table('./data/test.feature.txt', nrows=nrows)
y_test = pd.read_table('./data/test.txt', nrows=nrows)

c_params = [0.1, 1.0, 10.0]
n_layers = [(50,), (100,), (500,)]
result_df = pd.DataFrame(columns=["MODEL", "PARAMETER", "ACCURACY"])

for c in c_params:
    print(f"LR, C:{c}")
    model = LogisticRegression(C=c, solver='sag', random_state=1234)
    model.fit(X_train.values, y_train["CATEGORY"])
    result_series = pd.Series(
        [
            "LOGISTIC_REGRESSION", c, accuracy_score(y_test["CATEGORY"], model.predict(X_test.values))
        ], 
        index=result_df.columns
        )
    print(result_series)
    result_df = result_df.append(result_series, ignore_index=True)

for n_layer in n_layers:
    print(f"MLP, hidden_layer_sizes:{n_layer}")
    model = MLPClassifier(hidden_layer_sizes=n_layer, random_state=1234)
    model.fit(X_train.values, y_train["CATEGORY"])
    result_series = pd.Series(
        [
            "MLP", n_layer, accuracy_score(y_test["CATEGORY"], model.predict(X_test.values))
        ], 
        index=result_df.columns
        )
    print(result_series)
    result_df = result_df.append(result_series, ignore_index=True)

print(result_df.sort_values(by="ACCURACY", ascending=False))