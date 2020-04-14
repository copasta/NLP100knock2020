import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

X_train = pd.read_table('./data/train.feature.txt')
y_train = pd.read_table('./data/train.txt')

model = LogisticRegression(C=0.1, solver='lbfgs', random_state=1234)
model.fit(X_train.values, y_train["CATEGORY"])
joblib.dump(model, './data/model_ch06.joblib')