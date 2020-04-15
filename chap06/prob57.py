import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score

X_test = pd.read_table('./data/test.feature.txt')
y_test = pd.read_table('./data/test.txt')

model = joblib.load("./data/model_ch06.joblib")

for c_name, c_weight in zip(model.classes_, model.coef_):
    print(c_name)
    dict_coef = dict(zip(X_test.columns, c_weight))
    top_10 = sorted(dict_coef.items(), key=lambda x: x[1], reverse=True)[:10]
    bottom_10 = sorted(dict_coef.items(), key=lambda x: x[1], reverse=False)[:10]
    print("---TOP10---")
    print(top_10)
    print("---BOTTOM10---")
    print(bottom_10)