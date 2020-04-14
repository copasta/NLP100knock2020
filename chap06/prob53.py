import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

X_train = pd.read_table('./data/train.feature.txt')
y_train = pd.read_table('./data/train.txt')

model = joblib.load("./data/model_ch06.joblib")
# 予測確率
y_pred_proba = model.predict_proba(X_train.values)
# 予測カテゴリ
y_pred = model.predict(X_train.values)